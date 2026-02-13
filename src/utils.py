import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ==============================
# LOAD TRAINED MODEL
# ==============================
def load_trained_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model


# ==============================
# PREPROCESS FRAME
# ==============================
def preprocess_frame(frame):
    img = cv2.resize(frame, (224, 224))  # MUST match model input
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img, dtype=np.float32)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


# ==============================
# PREDICT STATE
# ==============================
def predict_state(model, frame):
    processed = preprocess_frame(frame)

    prediction = model.predict(processed, verbose=0)[0][0]
    confidence = float(prediction)

    if prediction > 0.5:
        state = "sleepy"
    else:
        state = "awake"
        confidence = 1 - confidence

    return state, confidence
