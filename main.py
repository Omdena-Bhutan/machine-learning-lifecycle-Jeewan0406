import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from src.data.data_loader import DataLoader
from src.data.data_cleaning import DataCleaning

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

@hydra.main(version_base=None, config_path='.', config_name='params')
def main(cfg: DictConfig):
    
    logging.info(f'Calling the functions from {DataLoader} class')
    
    try:
        # Create DataLoader instance from Hydra config
        data_loader = DataLoader.hydra_config(cfg)
        
        # Load the dataset
        df = data_loader.load_dataset()
        
        # Data cleaning
        dest_dir = Path(cfg.data.save_data.dest_path)
        output_path = dest_dir/"cleaned_data.csv"
        
        cleaner = DataCleaning(df = df, target_path = output_path)
        cleaner.saved_processed_data()
        logger.info('Pipeline executed successfully')
    
        
    except Exception as e:
        logging.error(f'Pipeline failed to execute: {e}')
        raise
        
if __name__ == '__main__':
    main()