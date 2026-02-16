# Standard library imports
import sys
import logging
from pathlib import Path
# Third party libraries and modules
from omegaconf import DictConfig, OmegaConf

def setup_logging(config: DictConfig) -> logging.Logger:
    if 'logging' not in config:
        raise ValueError(
            "Config must contain a 'logging' section for setup. "
            "Make sure you're passing config.yaml, not params.yaml."
        )
    
    log_config = config.logging
    
    # Create log directory
    log_path = Path(log_config.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get log level
    log_level = getattr(logging, log_config.level.upper())
    
    # Create formatter
    formatter = logging.Formatter(
        fmt=log_config.format,
        datefmt=log_config.date_format
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(log_config.log_file)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Console handler
    if log_config.show_in_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # Create and return logger
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured: {log_config.log_file}")
    logger.info("ConfigManager ready")
    
    return logger