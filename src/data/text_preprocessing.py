# text_cleaning.py (NEW FILE - put this in src/ folder)

import re
from pathlib import Path
import logging
from typing import Optional

import pandas as pd
from bs4 import BeautifulSoup


class TextPreprocessor:

    def __init__(
        self,
        input_path: Path | str,
        output_path: Path | str,
        text_column: str = "review",
        logger: Optional[logging.Logger] = None
    ):
        
        # Use provided logger or create fallback
        self.logger = logger or logging.getLogger(self.__class__.__name__)

        # Convert paths
        input_path = Path(input_path)
        self.output_path = Path(output_path)

        # Validate input file exists
        if not input_path.exists():
            self.logger.error(f"Input file not found: {input_path}")
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Load data
        try:
            df = pd.read_csv(input_path)
        except Exception as e:
            self.logger.error(f"Failed to read CSV: {e}")
            raise ValueError(f"Failed to read CSV file: {e}")

        # Validate DataFrame
        if not isinstance(df, pd.DataFrame):
            self.logger.error(f"Expected pandas DataFrame, got {type(df).__name__}")
            raise TypeError(f"Expected DataFrame, got {type(df).__name__}")

        if df.empty:
            self.logger.error("Input DataFrame is empty")
            raise ValueError("DataFrame is empty")

        if text_column not in df.columns:
            self.logger.error(
                f"Column '{text_column}' not found. Available: {list(df.columns)}"
            )
            raise ValueError(f"Column '{text_column}' not found")

        self.df = df.copy()
        self.text_column = text_column

        self.logger.info(
            f"TextCleaning initialized – column: '{self.text_column}', "
            f"rows: {len(self.df):,}, output: {self.output_path}"
        )

    def lower(self) -> 'TextPreprocessor':
        """Convert text to lowercase."""
        self.logger.info(f"Converting '{self.text_column}' to lowercase")
        self.df[self.text_column] = self.df[self.text_column].str.lower()
        return self

    def remove_html(self) -> 'TextPreprocessor':
        """Remove HTML tags."""
        self.logger.info(f"Removing HTML tags from '{self.text_column}'")

        def clean_html(text):
            if pd.isna(text):
                return ""
            try:
                soup = BeautifulSoup(str(text), "html.parser")
                return soup.get_text(separator=" ")
            except Exception as e:
                self.logger.warning(f"HTML parsing failed for a row: {e}")
                return str(text)

        self.df[self.text_column] = self.df[self.text_column].apply(clean_html)
        return self

    def remove_special_chars(self) -> 'TextPreprocessor':
        """Remove special characters, keep alphanumeric + whitespace."""
        self.logger.info(f"Removing special characters from '{self.text_column}'")
        self.df[self.text_column] = (
            self.df[self.text_column]
            .astype(str)
            .str.replace(r"[^\w\s']", '', regex=True)
        )
        return self

    def normalize_whitespace(self) -> 'TextPreprocessor':
        """Collapse multiple whitespaces and strip."""
        self.logger.info(f"Normalizing whitespace in '{self.text_column}'")
        self.df[self.text_column] = (
            self.df[self.text_column]
            .astype(str)
            .str.replace(r'\s+', ' ', regex=True)
            .str.strip()
        )
        return self

    def run_all(self) -> 'TextPreprocessor':
        """Apply standard cleaning pipeline in recommended order."""
        self.logger.info("Running full cleaning pipeline")
        return (
            self
            .lower()
            .remove_html()
            .remove_special_chars()
            .normalize_whitespace()
        )

    def save(self, path: Optional[Path | str] = None) -> Path:
        """
        Save cleaned DataFrame to CSV.

        Args:
            path: Optional custom save path (uses output_path if None)

        Returns:
            Path object of saved file
        """
        save_path = Path(path) if path else self.output_path

        # Create parent directory if needed
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save to CSV
        self.df.to_csv(save_path, index=False, encoding="utf-8")
        size_kb = save_path.stat().st_size / 1024

        self.logger.info(
            f"Saved {len(self.df):,} rows to {save_path.resolve()} "
            f"({size_kb:.1f} KB)"
        )
        return save_path

    def get_dataframe(self) -> pd.DataFrame:
        """
        Get the cleaned DataFrame without saving.

        Returns:
            Cleaned pandas DataFrame
        """
        return self.df