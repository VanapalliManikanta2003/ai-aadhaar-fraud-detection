import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from fraud_detection.cnn_model import create_model

model = create_model()
model.load_weights("model/document_fraud_model.weights.h5")
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

test_gen = ImageDataGenerator(rescale=1.0 / 255)
test = test_gen.flow_from_directory(
    "dataset/test",
    target_size=(128, 128),
    color_mode="grayscale",
    batch_size=16,
    class_mode="binary",
    shuffle=False
)

loss, acc = model.evaluate(test)
print("✅ Test Loss:", loss)
print("✅ Test Accuracy:", acc)
