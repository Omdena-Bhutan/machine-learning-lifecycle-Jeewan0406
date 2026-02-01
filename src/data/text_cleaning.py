# src/data/text_cleaning.py
import logging
import re
from pathlib import Path
from typing import Union, Optional

import pandas as pd
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class TextCleaning:
    """
    Clean text data in a pandas DataFrame column with method chaining.

    Example:
        cleaner = TextCleaning(df, output_path=Path("cleaned.csv"), text_column="review")
        cleaner.lower().remove_html().remove_special_chars().normalize_whitespace().save()
    """

    def __init__(
        self,
        df: pd.DataFrame,
        output_path: Union[Path, str],
        text_column: str = "review"
    ):
        # Input validation
        if not isinstance(df, pd.DataFrame):
            logger.error(f"Expected pandas DataFrame, got {type(df).__name__}")
            raise TypeError(f"Expected DataFrame, got {type(df).__name__}")

        if df.empty:
            logger.error("Input DataFrame is empty")
            raise ValueError("DataFrame is empty")

        if text_column not in df.columns:
            logger.error(f"Column '{text_column}' not found. Available: {list(df.columns)}")
            raise ValueError(f"Column '{text_column}' not found")

        self.df = df.copy()                     # safe copy
        self.text_column = text_column
        self.output_path = Path(output_path)    # always Path object

        logger.info(f"TextCleaning initialized - column: '{self.text_column}', "
                    f"rows: {len(self.df):,}, output: {self.output_path}")

    def lower(self) -> 'TextCleaning':
        """Convert text to lowercase"""
        logger.info(f"Converting '{self.text_column}' to lowercase")
        self.df[self.text_column] = (
            self.df[self.text_column]
            .fillna('')
            .astype(str)
            .str.lower()
        )
        return self

    def remove_html(self) -> 'TextCleaning':
        """Remove HTML tags"""
        logger.info(f"Removing HTML tags from '{self.text_column}'")

        def clean_html(text):
            if pd.isna(text) or text == '':
                return ""
            try:
                soup = BeautifulSoup(str(text), "html.parser")
                return soup.get_text(separator=" ")
            except Exception as e:
                logger.warning(f"HTML parsing failed for a row: {e}")
                return str(text)

        self.df[self.text_column] = self.df[self.text_column].apply(clean_html)
        return self

    def remove_special_chars(self) -> 'TextCleaning':
        """Remove special characters, keep alphanumeric + whitespace"""
        logger.info(f"Removing special characters from '{self.text_column}'")
        self.df[self.text_column] = (
            self.df[self.text_column]
            .fillna('')
            .astype(str)
            .str.replace(r'[^\w\s]', '', regex=True)
        )
        return self

    def normalize_whitespace(self) -> 'TextCleaning':
        """Collapse multiple whitespaces and strip"""
        logger.info(f"Normalizing whitespace in '{self.text_column}'")
        self.df[self.text_column] = (
            self.df[self.text_column]
            .fillna('')
            .astype(str)
            .str.replace(r'\s+', ' ', regex=True)
            .str.strip()
        )
        return self

    def run_all(self) -> 'TextCleaning':
        """Apply standard cleaning pipeline in recommended order"""
        logger.info("Running full cleaning pipeline")
        return (
            self
            .lower()
            .remove_html()
            .remove_special_chars()
            .normalize_whitespace()
        )

    def save(self, path: Optional[Union[Path, str]] = None) -> Path:
        """Save the cleaned DataFrame"""
        save_path = Path(path) if path else self.output_path

        save_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self.df.to_csv(save_path, index=False, encoding='utf-8')
            size_kb = save_path.stat().st_size / 1024
            logger.info(f"Saved {len(self.df):,} rows to {save_path} "
                        f"({size_kb:.1f} KB)")
            return save_path
        except Exception as e:
            logger.error(f"Failed to save: {e}", exc_info=True)
            raise