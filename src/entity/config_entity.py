# from dataclasses import dataclass

# @dataclass
# class DataIngestionConfig:
#     train_dir: str = "data/raw/fer2013/data/train"
#     test_dir: str = "data/raw/fer2013/data/test"
#     val_dir: str = "data/raw/fer2013/data/val"

# @dataclass
# class ModelTrainerConfig:
#     trained_model_file_path: str = "artifacts/emotion_model.keras"



from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    root_dir: str
    source_URL: str
    local_data_file: str
    unzip_dir: str

@dataclass
class PrepareBaseModelConfig:
    root_dir: str
    base_model_path: str
    updated_base_model_path: str
    params_image_size: list
    params_learning_rate: float
    params_include_top: bool
    params_weights: str
    params_classes: int

@dataclass
class TrainingConfig:
    root_dir: str
    trained_model_path: str
    updated_base_model_path: str
    training_data: str
    params_epochs: int
    params_batch_size: int
    params_is_augmentation: bool
    params_image_size: list

@dataclass
class EvaluationConfig:
    path_of_model: str
    training_data: str
    all_params: dict
    mlflow_uri: str
    params_image_size: list
    params_batch_size: int

# 👇 IMPORTANT
@dataclass
class ModelTrainerConfig:
    root_dir: str
    trained_model_file_path: str