import logging
from pathlib import Path
from typing import Optional

import mlflow
import mlflow.transformers
import numpy as np
from omegaconf import OmegaConf

from dataclasses import dataclass
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ───────────────── Configs ─────────────────

@dataclass
class ModelConfig:
    model_name: str
    num_labels: int
    freeze_base: bool
    cache_dir: Optional[str] = None


@dataclass
class PathConfig:
    output_dir: str


@dataclass
class TrainConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    logging_steps: int
    eval_steps: int
    save_steps: int
    early_stopping_patience: int
    experiment_name: str
    run_name: Optional[str] = None


# ───────────────── Model Class ─────────────────

class SentimentModel:
    def __init__(self, model_cfg: ModelConfig, path_cfg: PathConfig, train_cfg: TrainConfig):
        self.model_cfg = model_cfg
        self.path_cfg = path_cfg
        self.train_cfg = train_cfg

        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForSequenceClassification] = None
        self.trainer: Optional[Trainer] = None

        logger.info(f"Backbone: {self.model_cfg.model_name}")

    # ───────────────── Initialization ─────────────────

    def initialize_model(self):
        logger.info("Loading tokenizer + pretrained weights")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_cfg.model_name,
            cache_dir=self.model_cfg.cache_dir,
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_cfg.model_name,
            num_labels=self.model_cfg.num_labels,
            cache_dir=self.model_cfg.cache_dir,
        )

        if self.model is None:
            raise ValueError("Model failed to load")

        # Freeze backbone if requested
        if self.model_cfg.freeze_base:
            logger.info("FREEZING backbone -> training head only")

            # Freeze everything first (future proof)
            for p in self.model.parameters():
                p.requires_grad = False

            # Unfreeze classifier
            if hasattr(self.model, "pre_classifier"):
                for p in self.model.pre_classifier.parameters():
                    p.requires_grad = True

            for p in self.model.classifier.parameters():
                p.requires_grad = True

        else:
            logger.info("Full fine-tuning → all parameters trainable")

        # Log parameter stats
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        logger.info(
            f"Trainable params: {trainable:,} / {total:,} "
            f"({100*trainable/total:.2f}%)"
        )

    # ───────────────── Metrics ─────────────────

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)

        acc = accuracy_score(labels, preds)
        avg = "binary" if self.model_cfg.num_labels == 2 else "macro"

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average=avg, zero_division=0
        )

        return {
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    # ───────────────── Training ─────────────────

    def train(self, train_data: Dataset, val_data: Dataset):
        output_dir = Path(self.path_cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # GUARANTEE model exists
        if self.model is None:
            self.initialize_model()

        assert self.model is not None, "Model is None before Trainer initialization"

        mlflow.set_experiment(self.train_cfg.experiment_name)
        mlflow.transformers.autolog()

        args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.train_cfg.epochs,
            per_device_train_batch_size=self.train_cfg.batch_size,
            per_device_eval_batch_size=self.train_cfg.batch_size,
            learning_rate=self.train_cfg.learning_rate,
            weight_decay=self.train_cfg.weight_decay,
            warmup_steps=self.train_cfg.warmup_steps,
            logging_steps=self.train_cfg.logging_steps,
            eval_strategy="steps",         # ← NEW (correct)
            eval_steps=self.train_cfg.eval_steps,
            save_strategy="steps",
            save_steps=self.train_cfg.save_steps,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            run_name=self.train_cfg.run_name,
            report_to="mlflow",
            fp16=True,  # will be ignored on CPU; no crash
        )

        self.trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_data,
            eval_dataset=val_data,
            compute_metrics=self.compute_metrics,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=self.train_cfg.early_stopping_patience
                )
            ],
        )

        with mlflow.start_run(run_name=self.train_cfg.run_name):
            logger.info("Training started")
            self.trainer.train()
            logger.info("Training finished")

    # ───────────────── Saving ─────────────────

    def save_model(self, path: str):
        if self.trainer is None:
            logger.warning("Trainer not available — nothing to save")
            return

        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)

        self.trainer.save_model(str(p))
        if self.tokenizer:
            self.tokenizer.save_pretrained(str(p))
        
        logger.info(f"Model saved to {p}")

