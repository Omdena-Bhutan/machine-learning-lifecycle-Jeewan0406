import sys
import logging

# Local application imports
from src.utils.logging_setup import Logger
from src.utils.data_loader  import DataLoaderClass

from src.data.text_preprocessing import TextPreprocessor

def main():
    try:
        # Load configuration
        print("Loading configuration...")
        loader = DataLoaderClass("configs/logging.yaml")
        cfg = loader.load()
        
        params_loader = DataLoaderClass("params.yaml")
        params = params_loader.load()
        
        # Setup logging
        print("Setting up logging...")
        logger = Logger(cfg, experiment_name= params.experiment.name)
        
        
    
        # # Text preprocessing
        # logger.info("Running text preprocessing...")
        # cleaner = TextPreprocessor(
        #     input_path=cfg.path.raw.path,
        #     output_path=cfg.path.processed.path,
        #     text_column=params.data.text_column,
        #     logger=logger
        # )
        # cleaner.run_all().save()
        # logger.info("Text preprocessing completed successfully.")
        
    except Exception as e:
        if 'logger' in locals():
            logger.error(f"An error occurred: {e}")
        else:
            logging.error(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    main()