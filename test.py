import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Path to test dataset
test_dir = "dataset/test"

# Load saved model
model = tf.keras.models.load_model("models/mobilenetv2_sleep_model.h5")

# Image parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Test data generator
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# Evaluate model
loss, accuracy = model.evaluate(test_generator)

print("\nTest Loss:", loss)
print("Test Accuracy:", accuracy)