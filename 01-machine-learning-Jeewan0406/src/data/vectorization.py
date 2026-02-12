import logging
from pathlib import Path

import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer


logger = logging.getLogger(__name__)


class SentimentVectorization:
    """Handles sentiment data encoding and tokenization for model training."""

    def __init__(self,
                 input_path: Path,
                 output_path: Path,
                 model_name: str):
        """
        Initialize the vectorization pipeline.

        Args:
            input_path: Path to input CSV file with 'review' and 'sentiment' columns
            output_path: Path where processed data will be saved
            model_name: Pretrained model name for tokenizer
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.model_name = model_name
        self.tokenizer = None

        # Validate input file exists
        if not self.input_path.exists():
            logger.error(f"Input file {self.input_path} does not exist")
            raise FileNotFoundError(f"Input file {self.input_path} does not exist")
            # The line below will never be reached because of the raise above. It can be removed or moved if it was intended to log output directory creation.
            # logger.info(f"Creating output directory: {self.output_path}")

        # Create output directory if it doesn't exist at file specified by in the params.yaml
        # cfg.data.vectorization.path / "train_vectorized.csv" creates vectorization if there is only data but does not create train_vectroized.csv
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def load_and_encode_sentiment(self) -> pd.DataFrame:  # does not expect data, it expects file path, path already attached with object as per self.inputpath = inputpath
        """
        Load CSV data and encode sentiment labels to binary (0=negative, 1=positive).

        Returns:
            DataFrame with encoded sentiment column
        """
        logger.info(f"Loading data from {self.input_path}")

        try:
            df = pd.read_csv(self.input_path) # copies data from hard disk to RAM, creates new object called dataframe in RAM/changes will happen in RAM data
            logger.info(f"Loaded {len(df)} rows")
            logger.info(f"Sentiment distribution:\n{df['sentiment'].value_counts()}")

        except Exception as e:
            logger.error(f"Error reading input file: {e}")
            raise

        # Validate required columns
        if 'sentiment' not in df.columns or 'review' not in df.columns:
            raise ValueError("Input CSV must contain 'sentiment' and 'review' columns")

        # Encode sentiment to binary
        sentiment_map = {'negative': 0, 'positive': 1}
        df['label'] = df['sentiment'].map(sentiment_map)
        df['label'] = df['label'].astype(int)

        logger.info("Sentiment encoding complete:")
        logger.info(f"\n{df['label'].value_counts()}")

        print("\nOriginal sample:")
        print(df[['review', 'sentiment', 'label']].head())

        return df # return memory address where df object lives right now/ not copy of actual data like pointer (fun--> address in RAM)

# the DataFrame is in RAM, ut a function cannot use it unless it receives a reference
    def tokenize_function(self, df_to_tokenize: pd.DataFrame) -> Dataset: # needs result of the read csv file
        """
        Tokenize text examples for model input.

        Args:
            df_to_tokenize: DataFrame with 'review' and 'label' columns

        Returns:
            Tokenized HuggingFace Dataset
        """

        if self.tokenizer is None:
            logger.info(f"Loading tokenizer: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Convert to HuggingFace Dataset
        hf_dataset = Dataset.from_pandas(df_to_tokenize[['review', 'label']])

        data = hf_dataset.map(
            lambda examples: self.tokenizer(examples["review"], padding="max_length", truncation=True, max_length=512),
            batched=True,
            remove_columns=['review']
        )


        print("\nColumns after tokenization:")
        print(data.column_names)

        print("\nTokenized sample:")
        print(data[0])

        print("\nFirst 20 input_ids:")
        print(data[0]["input_ids"][:20])

        print("\nDecoded back:")
        print(self.tokenizer.decode(data[0]["input_ids"]))

        print("\nOriginal review:")
        print(df_to_tokenize["review"].iloc[0])


        return data

# Hugging Face Dataset, which internally stores data in the Apache Arrow format.
    def save_processed_data(self, dataset: Dataset):
      logger.info(f"Saving Arrow dataset to {self.output_path}")
      dataset.save_to_disk(str(self.output_path))
      logger.info("Arrow save complete")

    def run_pipeline(self) -> Dataset:
        """
        Execute full vectorization pipeline.

        Returns:
            Tokenized dataset ready for training
        """
        # Load and encode
        df = self.load_and_encode_sentiment()


         # Prepare tokenized dataset
        tokenized_dataset = self.tokenize_function(df)
        # tokenized_dataset.head(10) # Removed this line which caused the error

        # Save processed CSV
        self.save_processed_data(tokenized_dataset)

        logger.info("Vectorization pipeline complete")
        return tokenized_dataset


