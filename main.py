# Standard library imports
import sys
import logging

# Local application imports
from src.data.data_loader import DataLoaderClass
from src.data.logging_setup import setup_logging
from src.data.text_preprocessing import TextPreprocessor

def main():
    try:
        # Load configuration
        print("Loading configuration...")
        data_loader = DataLoaderClass("configs/config.yaml")
        cfg = data_loader.load()

        # Setup logging
        print("Setting up logging...")
        logger = setup_logging(cfg)
        logger.info("Logging is set up successfully.")

        # Load params if text_column is in params.yaml
        params_loader = DataLoaderClass("params.yaml")
        params = params_loader.load()

        # Text preprocessing
        logger.info("Running text preprocessing...")
        cleaner = TextPreprocessor(
            input_path=cfg.path.raw.path,
            output_path=cfg.path.processed.path,
            text_column=params.data.text_column,
            logger=logger
        )
        cleaner.run_all().save()
        logger.info("Text preprocessing completed successfully.")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()


