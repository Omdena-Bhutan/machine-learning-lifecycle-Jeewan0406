import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
from omegaconf import OmegaConf # Added import for OmegaConf to load cfg

logger = logging.getLogger(__name__)

class TrainTestSplit:
    """
    Handles train/test splitting of tabular data with stratification support.

    Attributes:
        input_path (Path): Path to the input CSV file
        output_path (Path): Directory where split files will be saved
        test_size (float): Proportion of data to use for testing (0.0-1.0)
        random_state (int): Random seed for reproducible splits
        feature_column (str): Name of the feature column
        target_column (str): Name of the target column
        stratify (bool): Whether to perform stratified splitting
        val_size (float): Proportion of data to use for validation (0.0-1.0)
    """

    def __init__(self,
                 input_path: Path,
                 output_path: Path,
                 test_size: float,
                 random_state: int,
                 feature_column: str,
                 target_column: str,
                 stratify: bool = True,
                 val_size: float = 0.0): # Added val_size with a default of 0.0

        self.input_path = input_path
        self.output_path = output_path
        self.test_size = test_size
        self.random_state = random_state
        self.feature_column = feature_column
        self.target_column = target_column
        self.stratify = stratify
        self.val_size = val_size

    def split_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: # Corrected return type
        """
        Performs the train/validation/test split and saves results to CSV files.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (train_df, val_df, test_df)

        Raises:
            FileNotFoundError: If input file doesn't exist
            ValueError: If required columns are missing
            pd.errors.EmptyDataError: If CSV is empty
        """
        # Check if file exists
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file {self.input_path} not found")

        # Load CSV with error handling
        try:
            self.df = pd.read_csv(self.input_path)
        except pd.errors.EmptyDataError:
            logger.error(f"CSV file is empty: {self.input_path}")
            raise

        # Validate required columns exist
        required_cols = [self.feature_column, self.target_column]
        missing_cols = [col for col in required_cols if col not in self.df.columns]

        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}. "
                f"Available columns: {list(self.df.columns)}"
            )

        logger.info(f'Loaded dataset with {len(self.df):,} rows from {self.input_path}')

        # Prepare features and target
        X = self.df[self.feature_column]
        y = self.df[self.target_column]

        # First split: train_val and test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y if self.stratify else None,
            shuffle=True
        )

        # Second split: From train_val, split into train and val
        # Calculate new validation size relative to the remaining data
        if self.val_size > 0 and (1.0 - self.test_size) > 0: # Ensure remaining data is not zero to avoid division by zero
            relative_val_size = self.val_size / (1.0 - self.test_size)
            # Ensure relative_val_size does not exceed 1
            if relative_val_size >= 1.0:
                logger.warning(
                    f"Calculated relative_val_size ({relative_val_size:.2f}) is >= 1.0. "
                    f"Validation set will be empty or same as train_val. "
                    f"Consider adjusting test_size ({self.test_size:.2f}) or val_size ({self.val_size:.2f})."
                )
                X_train, X_val = X_train_val, pd.Series(dtype='object')
                y_train, y_val = y_train_val, pd.Series(dtype='object')
            else:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train_val,
                    y_train_val,
                    test_size=relative_val_size,
                    random_state=self.random_state,
                    stratify=y_train_val if self.stratify else None,
                    shuffle=True
                )
        else: # No separate validation set, all train_val becomes train
            X_train, X_val = X_train_val, pd.Series(dtype='object') # Create empty series for X_val
            y_train, y_val = y_train_val, pd.Series(dtype='object') # Create empty series for y_val

        # Reconstruct DataFrames
        self.df_train = pd.DataFrame({
            self.feature_column: X_train,
            self.target_column: y_train
        })
        self.df_val = pd.DataFrame({ # df_val was missing
            self.feature_column: X_val,
            self.target_column: y_val
        })
        self.df_test = pd.DataFrame({
            self.feature_column: X_test,
            self.target_column: y_test
        })

        logger.info(
            f"Split completed -> train: {len(self.df_train):,} rows | "
            f"validation: {len(self.df_val):,} rows | " # Added validation rows
            f"test: {len(self.df_test):,} rows"
        )

        # Ensure output directory exists
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Save files
        train_path = self.output_path / "train.csv"
        val_path = self.output_path / "val.csv" # val_path
        test_path = self.output_path / "test.csv"

        self.df_train.to_csv(train_path, index=False)
        if len(self.df_val) > 0: # Only save if validation set is not empty
            self.df_val.to_csv(val_path, index=False)
        self.df_test.to_csv(test_path, index=False)


        logger.info(f"Saved -> {train_path}")
        if len(self.df_val) > 0: # Only log if validation set is not empty
            logger.info(f"Saved -> {val_path}")
        logger.info(f"Saved -> {test_path}")

        return self.df_train, self.df_val, self.df_test # Corrected return values