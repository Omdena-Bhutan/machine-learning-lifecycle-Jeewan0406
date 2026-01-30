import logging
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class DataLoader():
    
    def __init__(self, data_path: Path):
        self.data_path = data_path
        
        # Checks if the data_path exists 
        if not self.data_path.exists():
            logging.error(f'File not found at {self.data_path}')
            raise FileNotFoundError(f'File not found at {self.data_path}')
        
        # Checks if the path is file.
        if not self.data_path.is_file():
            logging.error(f'Path exists and is not a file {self.data_path}')
            raise ValueError(f'Path exists and is not a file {self.data_path}')
        
    @classmethod
    def hydra_config(cls, cfg) -> "DataLoader":  # returns the same class object
        """
        Create DataLoader instance from Hydra config
        """
        if not hasattr(cfg.data.raw, "raw_path"):
            logging.error('File path missing at data.raw.raw_path')
            raise ValueError('File path missing at data.raw.raw_path')
        
        data_path = Path(cfg.data.raw.raw_path)
        
        # Often Hydra paths are relative → resolve against original working directory
        if not data_path.is_absolute():
            from hydra.utils import get_original_cwd
            data_path = Path(get_original_cwd()) / data_path
            
        logger.debug(f"Resolved data path from Hydra: {data_path}")

        return cls(data_path)
    
    
    def load_dataset(self):
        """
        The function that loads the data after file configuration completes
        
        Args:
            None
        
        Returns: 
            pd.DataFrame
            
        Raises:
            Exception: If CSV loading fails
        """
        logger.info(f'Loading data from file path {self.data_path}')
        
        try:
            df = pd.read_csv(self.data_path)
            logger.info(f'File loaded successfully: {len(df)} rows, {len(df.columns)} columns')
            return df
        
        except Exception as e:
            logger.error(f'Failed to load csv file at {self.data_path}: {e}')
            raise
        
    def save_dataset(self, df: pd.DataFrame, path: Path = None) -> Path:
        """
        Save a DataFrame to CSV.
        If no path is provided, saves to the original data_path (overwrite).
        
        Args:
            df: DataFrame to save
            path: Optional target path
            
        Returns:
            Path: The path where data was saved
        """
        target_path = path if path is not None else self.data_path

        # Ensure parent directory exists
        target_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving dataset to {target_path}")
        try:
            df.to_csv(target_path, index=False)
            logger.debug("Save completed successfully")
            return target_path
        
        except Exception as e:
            logger.error(f"Failed to save CSV to {target_path}: {e}")
            raise