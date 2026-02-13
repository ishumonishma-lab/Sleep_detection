import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load trained model
model = tf.keras.models.load_model("models/mobilenetv2_sleep_model.h5")

# Path to image you want to test
image_path = "test_image.jpg"   # Change this to your image path

# Read image
image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found.")
    exit()

# Resize to MobileNetV2 input size
image_resized = cv2.resize(image, (224, 224))

# Convert BGR to RGB
image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

# Expand dimensions
image_array = np.expand_dims(image_rgb, axis=0)

# Preprocess for MobileNetV2
image_array = preprocess_input(image_array)

# Predict
prediction = model.predict(image_array)[0][0]

# Output result
if prediction > 0.5:
    print("Prediction: Sleepy 😴")
else:
    print("Prediction: Awake 👀")

print("Confidence:", prediction)