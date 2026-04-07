# import sys
# import tensorflow as tf

# from src.exception import CustomException
# from src.logger import logging


# class DataTransformation:
#     def __init__(self):
#         pass

#     def get_data_generators(self, train_dir, test_dir, val_dir):
#         try:
#             logging.info("Creating ImageDataGenerators")

#             train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
#                 rescale=1./255,
#                 rotation_range=20,
#                 zoom_range=0.2,
#                 width_shift_range=0.2,
#                 height_shift_range=0.2,
#                 horizontal_flip=True
#             )

#             test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
#                 rescale=1./255
#             )

#             train_generator = train_datagen.flow_from_directory(
#                 train_dir,
#                 target_size=(48, 48),
#                 batch_size=16,
#                 color_mode="grayscale",
#                 class_mode="categorical"
#             )

#             val_generator = test_datagen.flow_from_directory(
#                 val_dir,
#                 target_size=(48, 48),
#                 batch_size=16,
#                 color_mode="grayscale",
#                 class_mode="categorical"
#             )

#             test_generator = test_datagen.flow_from_directory(
#                 test_dir,
#                 target_size=(48, 48),
#                 batch_size=16,
#                 color_mode="grayscale",
#                 class_mode="categorical"
#             )

#             logging.info("Image generators created successfully")

#             return train_generator, val_generator, test_generator

#         except Exception as e:
#             raise CustomException(e, sys)


import yaml
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_params(params_path="params.yaml"):
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


class DataTransformation:
    def __init__(self, params_path="params.yaml"):
        self.params = load_params(params_path)

    def get_data_generators(self):
        data_params = self.params["data"]
        train_params = self.params["training"]

        image_size = train_params["image_size"]
        batch_size = train_params["batch_size"]
        seed = train_params["seed"]

        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=10,
            zoom_range=0.1,
            horizontal_flip=True
        )

        test_datagen = ImageDataGenerator(rescale=1.0 / 255)

        train_generator = train_datagen.flow_from_directory(
            data_params["train_dir"],
            target_size=(image_size, image_size),
            batch_size=batch_size,
            color_mode="grayscale",
            class_mode="categorical",
            shuffle=True,
            seed=seed
        )

        val_generator = test_datagen.flow_from_directory(
            data_params["val_dir"],
            target_size=(image_size, image_size),
            batch_size=batch_size,
            color_mode="grayscale",
            class_mode="categorical",
            shuffle=False
        )

        test_generator = test_datagen.flow_from_directory(
            data_params["test_dir"],
            target_size=(image_size, image_size),
            batch_size=batch_size,
            color_mode="grayscale",
            class_mode="categorical",
            shuffle=False
        )

        return train_generator, val_generator, test_generator