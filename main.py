import sys
import logging

# Local application imports
from src.utils.logging_setup import Logger
from src.utils.data_loader  import DataLoaderClass

from src.data.text_preprocessing import TextPreprocessor
from src.data.train_val_test_split import TrainTestSplit

def main():
    try:
        # Load configuration for logging
        print("Loading configuration...")
        loader = DataLoaderClass("configs/logging.yaml")
        c = loader.load()
        
        # Load parameters for the experiment
        print("Loading parameters...")
        params_loader = DataLoaderClass("params.yaml")
        params = params_loader.load()
        
        # Setup logging
        print("Setting up logging...")
        logger_instance = Logger(c, experiment_name=params.experiment.name)
        logger = logger_instance.get_logger() 
        
        # Load the config for data paths
        logger.info("Loading data paths configuration...")
        config_loader = DataLoaderClass("configs/config.yaml")
        cfg = config_loader.load()
        
        
        
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
        
        # Splitting data into Train, Valid and Test format
        logger.info("Splitting data into train, valid and test...")
        splitter = TrainTestSplit(
            config=cfg,
            params=params,
            logger=logger
        )
        train_path, valid_path, test_path = splitter.split_data()

        logger.info(f"Train path: {train_path}")
        logger.info(f"Valid path: {valid_path}")
        logger.info(f"Test path:  {test_path}")
        logger.info("Data splitting completed successfully.")
        
    except Exception as e:
        if 'logger' in locals():
            logger.error(f"An error occurred: {e}")
        else:
            logging.error(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    main()