import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from omegaconf import DictConfig


class TrainTestSplit:

    def __init__(
        self,
        config: DictConfig,
        params: DictConfig,
        logger: logging.Logger
    ):
        self.logger = logger

        self.test_size      = params.split.test_size
        self.val_size       = params.split.val_size
        self.random_state   = params.split.random_state
       

        self.feature_column = params.data.text_column
        self.target_column  = params.data.target_column

        self.input_path = Path(config.path.processed.path)
        
        self.train_path = Path(config.split.train) 
        self.valid_path = Path(config.split.valid)
        self.test_path  = Path(config.split.test) 
        self.stratify  = params.split.stratify

    def split_data(self) -> Tuple[Path, Optional[Path], Path]:

        # Validate input file exists
        if not self.input_path.exists():
            self.logger.error(f"Input file not found: {self.input_path}")
            raise FileNotFoundError(f"Input file {self.input_path} not found")

        # Load data
        try:
            df = pd.read_csv(self.input_path)
            self.logger.info(f"Loaded {len(df):,} rows from {self.input_path}")
        except pd.errors.EmptyDataError:
            self.logger.error(f"CSV file is empty: {self.input_path}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to read CSV: {e}")
            raise

        # Validate required columns exist
        required_cols = [self.feature_column, self.target_column]
        missing_cols  = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            self.logger.error(f"Missing columns: {missing_cols}. Available: {list(df.columns)}")
            raise ValueError(f"Missing required columns: {missing_cols}. Available columns: {list(df.columns)}")

        # Log class distribution if stratifying
        if self.stratify:
            class_dist = df[self.target_column].value_counts()
            self.logger.info(f"Class distribution: {dict(class_dist)}")

        # === FIRST SPLIT: Separate test set ===
        train_temp_idx, test_idx = train_test_split(
            df.index,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=df[self.target_column] if self.stratify else None,
            shuffle=True
        )

        self.logger.info(f"First split temp: {len(train_temp_idx):,} rows, test: {len(test_idx):,} rows")

        # === SECOND SPLIT: Separate validation set (if needed) ===
        if self.val_size > 0:
            relative_val_size = self.val_size / (1.0 - self.test_size)

            self.logger.info(f"Creating validation set with relative size {relative_val_size:.2f}")

            train_idx, val_idx = train_test_split(
                train_temp_idx,
                test_size=relative_val_size,
                random_state=self.random_state,
                stratify=df.loc[train_temp_idx, self.target_column] if self.stratify else None,
                shuffle=True
            )
        else:
            train_idx = train_temp_idx
            val_idx   = None
            self.logger.info("No validation set created (val_size=0)")

        # Log split summary
        self.logger.info("Split Summary:")
        self.logger.info(f"  Training:   {len(train_idx):,} rows ({len(train_idx)/len(df)*100:.1f}%)")
        if val_idx is not None:
            self.logger.info(f"  Validation: {len(val_idx):,} rows ({len(val_idx)/len(df)*100:.1f}%)")
        self.logger.info(f"  Test:       {len(test_idx):,} rows ({len(test_idx)/len(df)*100:.1f}%)")

        # === SAVE FILES ===
        self.train_path.parent.mkdir(parents=True, exist_ok=True)

        df.loc[train_idx].to_csv(self.train_path, index=False)
        self.logger.info(f"Saved training data {self.train_path}")

        if val_idx is not None:
            df.loc[val_idx].to_csv(self.valid_path, index=False)
            self.logger.info(f"Saved validation data {self.valid_path}")

        df.loc[test_idx].to_csv(self.test_path, index=False)
        self.logger.info(f"Saved test data {self.test_path}")

        return (
            self.train_path,
            self.valid_path if val_idx is not None else None,
            self.test_path
        )