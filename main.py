import logging
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from src.data.data_loader import DataLoader
from src.data.data_cleaning import DataCleaning
from src.data.text_cleaning import TextCleaning
from src.data.nlp_preprocessing import NLPProcessor

# ────────────────────────────────────────────────
# Logging setup (moved outside function for clarity)
# ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,                     # ← INFO is usually better for pipelines
    format='%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path=".", config_name="params")
def main(cfg: DictConfig):
    """
    Main pipeline entry point.
    Supports running specific stages via command line: python main.py stage=data_cleaning
    """
    # Get stage from Hydra override (stage=xxx or +stage=xxx both work)
    stage = cfg.get("stage", "full")   # default = full pipeline

    logger.info(f"Pipeline started — stage = {stage}")

    try:
        # ───────────────────────────────
        # 1. Always load the raw data
        # ───────────────────────────────
        logger.info("Loading raw dataset...")
        loader = DataLoader.hydra_config(cfg)
        df = loader.load_dataset()

        # ───────────────────────────────
        # 2. Structural / duplicate cleaning
        # ───────────────────────────────
        if stage in ("full", "data_loader", "data_cleaning"):
            logger.info("Running structural data cleaning...")
            cleaner = DataCleaning(
                df=df,
                target_path=Path(cfg.data.save_data.dest_path) / "cleaned_data.csv"
            )
            # FIXED: method name typo
            cleaner.saved_processed_data()   # ← this is the real name in your class

        # ───────────────────────────────
        # 3. Text cleaning
        # ───────────────────────────────
        if stage in ("full", "text_cleaning"):
            logger.info("Running text cleaning...")
            
            # Important improvement: use the already cleaned structural data
            # (instead of re-using the raw df)
            if stage == "text_cleaning":
                cleaned_path = Path(cfg.data.save_data.dest_path) / "cleaned_data.csv"
                if not cleaned_path.is_file():
                    raise FileNotFoundError(
                        f"Structural cleaned file not found: {cleaned_path}\n"
                        "Run the 'data_cleaning' stage first."
                    )
                logger.info(f"Loading structurally cleaned data from {cleaned_path}")
                df = pd.read_csv(cleaned_path)

            text_cleaner = TextCleaning(
                df=df,
                output_path=Path(cfg.data.text_data.text_path),
                text_column="review"   # ← make explicit (good practice)
            )
            text_cleaner.run_all().save()

        logger.info(f"Pipeline finished successfully — stage: {stage}")
        
        nlp = NLPProcessor(input_path = cfg.data.text_data.text_path,
                            output_path=cfg.data.nlp_data.nlp_path)
        nlp.run()
        

    except Exception as e:
        logger.exception("Pipeline failed")   # ← .exception() gives full traceback
        raise


if __name__ == "__main__":
    main()