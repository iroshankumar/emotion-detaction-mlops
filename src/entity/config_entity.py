from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_dir: str = "data/raw/fer2013/data/train"
    test_dir: str = "data/raw/fer2013/data/test"
    val_dir: str = "data/raw/fer2013/data/val"

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = "artifacts/emotion_model.keras"