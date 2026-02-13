Sleep Detection System using MobileNetV2

This project is a real-time sleep detection system developed using deep learning and computer vision.
The system identifies whether a person is awake or sleepy based on eye images captured through a webcam.
A MobileNetV2 transfer learning model is used for classification.

The main goal of this project is to build a lightweight and efficient drowsiness detection system that can work in real time.

Project Overview

The model is trained on eye images that are classified into two categories:

Awake

Sleepy

During real-time execution, the webcam captures the face, detects the eye region using OpenCV Haar cascade, and sends the eye image to the trained model for prediction.
The output is displayed on the screen along with a confidence score.

Model Details

Base model: MobileNetV2 (pretrained on ImageNet)

Frozen convolutional layers for feature extraction

Global Average Pooling layer

Dropout layer to reduce overfitting

Dense layer with sigmoid activation for binary classification

Input image size: 224 × 224
Loss function: Binary crossentropy
Optimizer: Adam

Folder Structure
Sleepy/
│
├── app/
│   └── live_sleep_detection.py
│
├── dataset/
│   ├── train/
│   │   ├── awake/
│   │   └── sleepy/
│   └── test/
│       ├── awake/
│       └── sleepy/
│
├── models/
│   └── mobilenetv2_sleep_model.h5
│
├── src/
│   └── utils.py
│
├── train.py
├── test.py
└── requirements.txt

Installation

Clone the repository:

git clone https://github.com/your-username/sleep-detection.git
cd sleep-detection


Install the required libraries:

pip install -r requirements.txt

Training the Model

Training is required only once, unless the dataset or model is modified.

python train.py


The trained model will be saved in the models folder.

Testing the Model

To evaluate model performance on the test dataset:

python test.py

Running Real-Time Detection

To start webcam-based detection:

python app/live_sleep_detection.py


Press Q to close the webcam window.

Output

The system displays:

Green bounding box for awake

Red bounding box for sleepy

Confidence value for the prediction

Technologies Used

Python
TensorFlow / Keras
OpenCV
NumPy
MobileNetV2

Future Improvements

Improve accuracy with more eye images
Add eye aspect ratio based detection
Optimize performance for low-end systems
Deploy as a web or mobile application

Author

Monishma
AI and Data Science
PSNA College of Engineering and Technology

This project was developed as part of academic work in deep learning and computer vision.
