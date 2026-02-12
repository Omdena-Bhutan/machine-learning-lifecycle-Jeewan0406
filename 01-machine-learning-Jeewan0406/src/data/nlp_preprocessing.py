import logging
from pathlib import Path
from typing import Optional

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

NLTK_DATA_DIR = Path(__file__).resolve().parents[2] / "nltk_data"
nltk.data.path = [str(NLTK_DATA_DIR)]
# -------------------------------
# NLTK resource bootstrap
# -------------------------------
def ensure_nltk_resources() -> None:
    resources = [
        "punkt",
        "punkt_tab",
        "wordnet",
        "stopwords",
    ]

    for res in resources:
        nltk.download(res, quiet=True, download_dir="./nltk_data")

ensure_nltk_resources()


# -------------------------------
# NLP Processor
# -------------------------------
class NLPProcessor:
    def __init__(
        self,
        input_path: str | Path,
        output_path: str | Path,
        text_column: str = "review",
        encoding: str = "utf-8",
    ):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.text_column = text_column
        self.encoding = encoding

        self.logger = logging.getLogger(__name__)

        self.df: Optional[pd.DataFrame] = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))

    # -------------------------------
    # Internal validation
    # -------------------------------
    def _validate_paths(self) -> None:
        """Validate input file and prepare output directory."""
        if not self.input_path.is_file():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    # -------------------------------
    # Pipeline steps
    # -------------------------------
    def load(self) -> None:
        self.logger.info("Loading data from %s", self.input_path)

        self.df = pd.read_csv(
            self.input_path,
            encoding=self.encoding,
            on_bad_lines="warn",
        )

        if self.text_column not in self.df.columns:
            raise ValueError(f"Column '{self.text_column}' not found")

        self.logger.info("Loaded %d rows", len(self.df))


    def tokenize(self) -> None:
        self.logger.info("Tokenizing text")

        self.df[self.text_column] = (
            self.df[self.text_column]
            .astype(str)
            .str.lower()
            .apply(word_tokenize)
        )

    def lemmatize(self) -> None:
        self.logger.info("Lemmatizing tokens")

        self.df[self.text_column] = self.df[self.text_column].apply(
            lambda tokens: [self.lemmatizer.lemmatize(t) for t in tokens]
        )

    def remove_stopwords(self) -> None:
        self.logger.info("Removing stopwords")

        self.df[self.text_column] = self.df[self.text_column].apply(
            lambda tokens: [t for t in tokens if t not in self.stop_words]
        )

    def save(self) -> None:
        self.logger.info("Saving output to %s", self.output_path)

        self.df.to_csv(
            self.output_path,
            index=False,
            encoding=self.encoding,
        )

    # -------------------------------
    # Orchestration
    # -------------------------------
    def run(self) -> None:
        self.logger.info("Starting NLP preprocessing pipeline")

        self._validate_paths()
        self.load()
        self.tokenize()
        self.lemmatize()
        self.remove_stopwords()
        self.save()

        self.logger.info("NLP preprocessing completed successfully")
