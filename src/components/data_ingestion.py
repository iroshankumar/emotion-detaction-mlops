import os
import sys

from src.entity.config_entity import DataIngestionConfig
from src.exception import CustomException
from src.logger import logging


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered data ingestion component")

        try:
            train_dir = self.ingestion_config.train_dir
            test_dir = self.ingestion_config.test_dir
            val_dir = self.ingestion_config.val_dir

            if not os.path.exists(train_dir):
                raise FileNotFoundError(f"Train directory not found: {train_dir}")

            if not os.path.exists(test_dir):
                raise FileNotFoundError(f"Test directory not found: {test_dir}")

            if not os.path.exists(val_dir):
                raise FileNotFoundError(f"Validation directory not found: {val_dir}")

            logging.info(f"Train directory found: {train_dir}")
            logging.info(f"Test directory found: {test_dir}")
            logging.info(f"Validation directory found: {val_dir}")

            return train_dir, test_dir, val_dir

        except Exception as e:
            raise CustomException(e, sys)