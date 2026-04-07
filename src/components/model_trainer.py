# # import os
# # import sys
# # import tensorflow as tf

# # from src.entity.config_entity import ModelTrainerConfig
# # from src.exception import CustomException
# # from src.logger import logging


# # class ModelTrainer:
# #     def __init__(self):
# #         self.model_trainer_config = ModelTrainerConfig()

# #     def build_model(self, num_classes):
# #         model = tf.keras.models.Sequential([
# #             tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
# #             tf.keras.layers.MaxPooling2D(2, 2),

# #             tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
# #             tf.keras.layers.MaxPooling2D(2, 2),

# #             tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
# #             tf.keras.layers.MaxPooling2D(2, 2),

# #             tf.keras.layers.Flatten(),
# #             tf.keras.layers.Dense(128, activation='relu'),
# #             tf.keras.layers.Dropout(0.5),
# #             tf.keras.layers.Dense(num_classes, activation='softmax')
# #         ])

# #         model.compile(
# #             optimizer='adam',
# #             loss='categorical_crossentropy',
# #             metrics=['accuracy']
# #         )

# #         return model

# #     def initiate_model_trainer(self, train_generator, val_generator):
# #         try:
# #             logging.info("Starting model training")

# #             model = self.build_model(num_classes=train_generator.num_classes)

# #             history = model.fit(
# #                 train_generator,
# #                 validation_data=val_generator,
# #                 epochs=1
# #             )

# #             os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)
# #             model.save(self.model_trainer_config.trained_model_file_path)

# #             logging.info(f"Model saved at {self.model_trainer_config.trained_model_file_path}")

# #             val_accuracy = history.history['val_accuracy'][-1]

# #             return val_accuracy

# #         except Exception as e:
# #             raise CustomException(e, sys)


# import os
# import json
# import yaml
# import tensorflow as tf
# import dagshub
# import mlflow
# import mlflow.keras
# from tensorflow.keras import layers, models
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


# def load_params(params_path="params.yaml"):
#     with open(params_path, "r") as f:
#         return yaml.safe_load(f)


# class ModelTrainer:
#     def __init__(self, params_path="params.yaml"):
#         self.params = load_params(params_path)

#     def build_model(self):
#         input_shape = tuple(self.params["model"]["input_shape"])
#         num_classes = self.params["model"]["num_classes"]
#         learning_rate = self.params["training"]["learning_rate"]

#         model = models.Sequential([
#             layers.Input(shape=input_shape),

#             layers.Conv2D(32, (3, 3), activation='relu'),
#             layers.MaxPooling2D(2, 2),

#             layers.Conv2D(64, (3, 3), activation='relu'),
#             layers.MaxPooling2D(2, 2),

#             layers.Conv2D(128, (3, 3), activation='relu'),
#             layers.MaxPooling2D(2, 2),

#             layers.Flatten(),
#             layers.Dense(128, activation='relu'),
#             layers.Dropout(0.5),
#             layers.Dense(num_classes, activation='softmax')
#         ])

#         model.compile(
#             optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
#             loss='categorical_crossentropy',
#             metrics=['accuracy']
#         )

#         return model

#     def train(self, train_generator, val_generator):
#         model = self.build_model()

#         epochs = self.params["training"]["epochs"]
#         model_path = self.params["artifacts"]["model_path"]
#         metrics_path = self.params["artifacts"]["metrics_path"]
#         history_path = self.params["artifacts"]["history_path"]

#         os.makedirs(os.path.dirname(model_path), exist_ok=True)
#         os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

#         callbacks = [
#             EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
#             ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
#             ModelCheckpoint(model_path, monitor="val_accuracy", save_best_only=True, verbose=1)
#         ]

#         history = model.fit(
#             train_generator,
#             validation_data=val_generator,
#             epochs=epochs,
#             callbacks=callbacks
#         )

#         val_loss, val_accuracy = model.evaluate(val_generator, verbose=0)

#         metrics = {
#             "val_loss": float(val_loss),
#             "val_accuracy": float(val_accuracy)
#         }

#         with open(metrics_path, "w") as f:
#             json.dump(metrics, f, indent=4)

#         with open(history_path, "w") as f:
#             json.dump(history.history, f, indent=4)

#         print(f"Training completed. Validation Accuracy: {val_accuracy:.4f}")
#         print(f"Model saved to: {model_path}")
#         print(f"Metrics saved to: {metrics_path}")
#         print(f"History saved to: {history_path}")

#         return model



import os
import json
import yaml
import tensorflow as tf
import dagshub
import mlflow
import mlflow.keras

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


def load_params(params_path="params.yaml"):
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


class ModelTrainer:
    def __init__(self, params_path="params.yaml"):
        self.params = load_params(params_path)

    def build_model(self):
        input_shape = tuple(self.params["model"]["input_shape"])
        num_classes = self.params["model"]["num_classes"]
        learning_rate = self.params["training"]["learning_rate"]

        model = models.Sequential([
            layers.Input(shape=input_shape),

            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),

            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),

            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),

            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def train(self, train_generator, val_generator):
        model = self.build_model()

        epochs = self.params["training"]["epochs"]
        model_path = self.params["artifacts"]["model_path"]
        metrics_path = self.params["artifacts"]["metrics_path"]
        history_path = self.params["artifacts"]["history_path"]

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
            ModelCheckpoint(model_path, monitor="val_accuracy", save_best_only=True, verbose=1)
        ]

        # DAGsHub + MLflow init
        dagshub.init(
            repo_owner="iroshankumar",
            repo_name="emotion-detaction-mlops",
            mlflow=True
        )

        with mlflow.start_run():
            mlflow.log_params({
                "image_size": self.params["training"]["image_size"],
                "batch_size": self.params["training"]["batch_size"],
                "epochs": self.params["training"]["epochs"],
                "learning_rate": self.params["training"]["learning_rate"],
                "num_classes": self.params["model"]["num_classes"]
            })

            history = model.fit(
                train_generator,
                validation_data=val_generator,
                epochs=epochs,
                callbacks=callbacks
            )

            val_loss, val_accuracy = model.evaluate(val_generator, verbose=0)

            metrics = {
                "val_loss": float(val_loss),
                "val_accuracy": float(val_accuracy)
            }

            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=4)

            with open(history_path, "w") as f:
                json.dump(history.history, f, indent=4)

            mlflow.log_metrics(metrics)
            mlflow.keras.log_model(model, "model")

            print(f"Training completed. Validation Accuracy: {val_accuracy:.4f}")
            print(f"Model saved to: {model_path}")
            print(f"Metrics saved to: {metrics_path}")
            print(f"History saved to: {history_path}")

        return model