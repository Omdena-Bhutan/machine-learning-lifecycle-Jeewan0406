# Standard library imports
import sys
import logging
from pathlib import Path
from datetime import datetime

# Third party libraries and modules
from omegaconf import DictConfig, OmegaConf


class Logger:
    
    def __init__(self, config: DictConfig, experiment_name: str = None):

        self.config = config
        self.log_config = config.logging
        self.experiment_name = experiment_name
        
        
        log_base = Path(self.config.logging.log_file).parent
        self.experiment_dir = (log_base/self.experiment_name/f"{self.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        
        self.logger = self._setup_logger()

        
    def _setup_logger(self) -> logging.Logger:
        log_level = getattr(logging, self.config.logging.level.upper())

        # Named experiment logger
        logger = logging.getLogger(self.experiment_name)
        logger.setLevel(log_level)

        # Guard: avoid duplicate handlers if logger already exists
        if logger.handlers:
            logger.handlers.clear()

        # Formatter from config
        formatter = logging.Formatter(
            fmt=self.log_config.format,
            datefmt=self.log_config.date_format
        )

        # File handler
        file_handler = logging.FileHandler(self.experiment_dir / "experiment.log")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Console handler — controlled by config flag
        if self.log_config.show_in_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(log_level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        return logger



