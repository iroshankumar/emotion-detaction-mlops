import os
import cv2
import numpy as np
import tensorflow as tf


class PredictPipeline:
    def __init__(self):
        self.model = tf.keras.models.load_model("artifacts/model/emotion_model.keras")
        self.emotion_labels = {
            0: "Angry",
            1: "Disgust",
            2: "Fear",
            3: "Happy",
            4: "Neutral",
            5: "Sad",
            6: "Surprise"
        }

    def preprocess_image(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (48, 48))
        img = img / 255.0
        img = np.reshape(img, (1, 48, 48, 1))
        return img

    def predict(self, image_path):
        processed_image = self.preprocess_image(image_path)
        predictions = self.model.predict(processed_image)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = float(np.max(predictions))
        emotion = self.emotion_labels[predicted_class]

        return emotion, confidence