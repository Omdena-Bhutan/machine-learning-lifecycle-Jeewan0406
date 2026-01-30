import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from src.data.data_loader import DataLoader

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
        
        # # Save the dataset to processed directory
        # save_path = Path(cfg.data.save_data.dest_path) / 'imdb_processed.csv'
        # data_loader.save_dataset(df, save_path)
        
        # logging.info('Pipeline completed successfully')
        
    except Exception as e:
        logging.error(f'Pipeline failed to execute: {e}')
        raise
        
if __name__ == '__main__':
    main()