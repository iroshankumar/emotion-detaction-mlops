# from src.components.data_ingestion import DataIngestion
# from src.components.data_transformation import DataTransformation
# from src.components.model_trainer import ModelTrainer
# from src.logger import logging

# if __name__ == "__main__":
#     logging.info("Training pipeline started")

#     ingestion = DataIngestion()
#     train_dir, test_dir, val_dir = ingestion.initiate_data_ingestion()

#     transformation = DataTransformation()
#     train_generator, val_generator, test_generator = transformation.get_data_generators(
#         train_dir, test_dir, val_dir
#     )

#     trainer = ModelTrainer()
#     val_accuracy = trainer.initiate_model_trainer(train_generator, val_generator)

#     print(f"Training completed. Validation Accuracy: {val_accuracy:.4f}")


from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


def main():
    data_transformation = DataTransformation()
    train_generator, val_generator, test_generator = data_transformation.get_data_generators()

    model_trainer = ModelTrainer()
    model_trainer.train(train_generator, val_generator)


if __name__ == "__main__":
    main()