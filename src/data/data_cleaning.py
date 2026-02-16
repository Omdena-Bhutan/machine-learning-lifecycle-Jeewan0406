
# Python native standard libraries and modules
import logging # Used to record runtime events for monitoring, debugging, and auditability
from pathlib import Path  # Used for highlevel path and directory path handling 
from typing import Optional 
from dataclasses import dataclass # Automates boilerplate generation (__init__, __repr__) for data-centric classes\\
    
# Third party imports
import pandas as pd # Used for data manipulation and analysis in tabular format



logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

@dataclass # This decorator tells Python to auto generate __init__, __repr__
class DataCleaning():
    """
    Handles data cleaning operations and saving of processed datasets.
    """
    df: pd.DataFrame
    target_path: Optional[Path] = None
    
    
    def drop_duplicate(self)->pd.DataFrame:
        """
        The logic to remove duplicate rows from the DataFrame.

        This method returns a new DataFrame with duplicate rows removed,
        based on all columns. The original DataFrame is not modified.

        Returns:
            pd.DataFrame: A DataFrame containing only unique rows.

        Notes:
            - Duplicate detection is performed across all columns.
            - The first occurrence of each duplicate row is retained.
            - This operation is non-destructive (no in-place modification).
        """
        logger.info("Dropping duplicate rows")
        return self.df.drop_duplicates()
    
    
    def saved_processed_data(self)->Path:
        """
        Save the cleaned dataset to the target path as a CSV file.

        This method removes duplicate rows from the input DataFrame using
        `drop_duplicate()`, ensures the parent directory of the target path
        exists, and writes the cleaned data to disk in CSV format.

        Returns:
            Path: The file system path where the cleaned CSV file was saved.

        Raises:
            ValueError: If `target_path` is not set on the instance.
            Exception: If writing the CSV file to disk fails.

        Notes:
            - The output file is written without the DataFrame index.
            - Parent directories are created automatically if they do not exist.
            - This method does not modify the original DataFrame in place.
        """
        
        cleaned_df = self.drop_duplicate()
        
        if self.target_path is None: 
            raise ValueError(f'No target at {self.target_path}')
        
        self.target_path.parent.mkdir(parents = True, exist_ok=True)            
        
        logger.info(f'Save data to {self.target_path}')
        
        try: 
            # to_csv turns table into comma-separated
            cleaned_df.to_csv(self.target_path, index = False) 
            logger.debug("Save completed successfully")
            return self.target_path
        
        except Exception as e:
            logger.error(f'Failed to save csv to {self.target_path}{e}')
            raise
        
        
        
        
        
    
    

