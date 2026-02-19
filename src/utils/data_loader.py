# Standard library
import sys
import logging
from pathlib import Path
# Third party libraries and modules
import pandas as pd
from omegaconf import DictConfig, OmegaConf

class DataLoaderClass:
    def __init__(self, path: str | Path):
        self.path = Path(path)
    
    def load(self) -> DictConfig | pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(f"File not found: {self.path}")
        
        ext = self.path.suffix.lower()
        
        if ext in ['.yaml', '.yml']:
            return self._load_yaml()
        elif ext in ['.csv']:
            return self._load_csv()
        else:
            raise ValueError(
                f"Unsupported file type '{ext}'. "
                f"Supported types: .yaml, .yml, .csv"
            )
    
    def _load_yaml(self) -> DictConfig:
        try:
            return OmegaConf.load(self.path)
        except Exception as e:
            raise ValueError(f"Error loading YAML file: {e}")
    
    def _load_csv(self) -> pd.DataFrame:
        try:
            return pd.read_csv(self.path)
        except Exception as e:
            raise ValueError(f"Error loading CSV file: {e}")