import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from fraud_detection.cnn_model import create_model
import os

DATASET_PATH = "dataset_all"
MODEL_DIR = "model"
WEIGHTS_PATH = "model/document_fraud_model.weights.h5"

os.makedirs(MODEL_DIR, exist_ok=True)

datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=10,
    zoom_range=0.1
)

train = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(128, 128),
    color_mode="grayscale",
    batch_size=16,
    class_mode="binary",
    subset="training"
)

val = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(128, 128),
    color_mode="grayscale",
    batch_size=16,
    class_mode="binary",
    subset="validation"
)

model = create_model()
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(train, validation_data=val, epochs=10)

model.save_weights(WEIGHTS_PATH)
print("✅ Model weights saved at:", WEIGHTS_PATH)
