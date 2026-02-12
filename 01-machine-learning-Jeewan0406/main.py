import os
import logging
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig
from datasets import load_from_disk, Dataset

from src.data.split import TrainTestSplit
from src.data.data_loader import DataLoader
from src.data.data_cleaning import DataCleaning
from src.data.text_cleaning import TextCleaning
from src.data.vectorization import SentimentVectorization
from src.model.model import SentimentModel, ModelConfig, PathConfig, TrainConfig

# ────────────────────────────────────────────────
# Logging setup
# ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path=".", config_name="params")
def main(cfg: DictConfig):
    """
    Main pipeline entry point.
    Supports running specific stages via command line: python main.py +stage=data_loader
    """
    stage = cfg.get("stage", "full")
    logger.info(f"Pipeline started - stage = {stage}")

    try:
        # ───────────────────────────────
        # 1. Load raw data
        # ───────────────────────────────
        logger.info("Loading raw dataset...")
        loader = DataLoader.hydra_config(cfg)
        df = loader.load_dataset()

        # ───────────────────────────────
        # 2. Structural/duplicate cleaning
        # ───────────────────────────────
        if stage in ("full", "data_loader", "data_cleaning"):
            logger.info("Running structural data cleaning...")
            cleaner = DataCleaning(
                df=df,
                target_path=Path(cfg.data.save_data.dest_path) / "cleaned_data.csv"
            )
            cleaner.saved_processed_data()

        # ───────────────────────────────
        # 3. Text cleaning
        # ───────────────────────────────
        if stage in ("full", "text_cleaning"):
            logger.info("Running text cleaning...")
            
            if stage == "text_cleaning":
                cleaned_path = Path(cfg.data.save_data.dest_path) / "cleaned_data.csv"
                if not cleaned_path.is_file():
                    raise FileNotFoundError(
                        f"Structural cleaned file not found: {cleaned_path}\n"
                        "Run the 'data_cleaning' stage first."
                    )
                logger.info(f"Loading structurally cleaned data from {cleaned_path}")
                df = pd.read_csv(cleaned_path)

            text_cleaner = TextCleaning(
                df=df,
                output_path=Path(cfg.data.text_data.text_path),
                text_column="review"
            )
            text_cleaner.run_all().save()
            
        # ───────────────────────────────
        # 4. Data splitting
        # ───────────────────────────────
        logger.info("Splitting data...")
        
        # Create output directory
        output_dir = Path(cfg.data.split.train_path).parent 
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize splitter with output_path
        train_test_splitter = TrainTestSplit(
            input_path=Path(cfg.data.cleaned.cleaned_path),
            output_path=output_dir,  # ✅ Added required output_path parameter
            test_size=cfg.data.split.test_size,
            val_size=cfg.data.split.val_size,
            random_state=cfg.data.split.random_state,
            feature_column=cfg.data.split.feature_column,
            target_column=cfg.data.split.target_column,
            stratify=cfg.data.split.stratify
        )

        # Split the data
        train_df, val_df, test_df = train_test_splitter.split_data()

        logger.info("Data splitting completed successfully.")
        logger.info(f"Train set: {len(train_df)} rows")
        logger.info(f"Validation set: {len(val_df)} rows")
        logger.info(f"Test set: {len(test_df)} rows")
        
        # ───────────────────────────────
        # 5. Vectorization
        # ───────────────────────────────
        logger.info("Vectorization stage")
        
        train_vectorizer = SentimentVectorization(
            input_path=Path(cfg.data.split.train_path),
            output_path=Path(cfg.data.vectorization.path) / "train_vectorized",
            model_name=cfg.data.vectorization.model_name
        )
        train_vectorizer.run_pipeline()
        
        val_vectorizer = SentimentVectorization(
            input_path=Path(cfg.data.split.val_path),
            output_path=Path(cfg.data.vectorization.path) / "val_vectorized",
            model_name=cfg.data.vectorization.model_name
        )
        val_vectorizer.run_pipeline()
        
        logger.info("Training and validation vectorization completed successfully.")
        
        # ───────────────────────────────
        # 6. Model Training
        # ───────────────────────────────
        logger.info("Starting model training...")
        
        # Create configs
        model_cfg = ModelConfig(
            model_name=cfg.model.model_name,
            num_labels=cfg.model.num_labels,
            freeze_base=cfg.model.freeze_base,
            cache_dir=cfg.model.cache_dir
        )

        path_cfg = PathConfig(output_dir=cfg.path.output_dir)

        train_cfg = TrainConfig(
            epochs=cfg.train.epochs,
            batch_size=cfg.train.batch_size,
            learning_rate=cfg.train.learning_rate,
            weight_decay=cfg.train.weight_decay,
            warmup_steps=cfg.train.warmup_steps,
            logging_steps=cfg.train.logging_steps,
            eval_steps=cfg.train.eval_steps,
            save_steps=cfg.train.save_steps,
            early_stopping_patience=cfg.train.early_stopping_patience,
            experiment_name=cfg.train.experiment_name,
            run_name=cfg.train.run_name,
        )

        # Load vectorized datasets
        logger.info("Loading vectorized datasets...")
        train_ds = load_from_disk(str(Path(cfg.data.vectorization.path) / "train_vectorized"))
        val_ds = load_from_disk(str(Path(cfg.data.vectorization.path) / "val_vectorized"))

        # Train model
        model = SentimentModel(model_cfg, path_cfg, train_cfg)
        model.train(train_ds, val_ds)
        
        # Save model
        save_path = str(Path(cfg.path.output_dir) / "best_model")
        model.save_model(save_path)
        
        logger.info("Model training completed successfully.")
        logger.info(f"Model saved to: {save_path}")
            
    except Exception as e:
        logger.exception("Pipeline failed")
        raise

if __name__ == "__main__":
    main()