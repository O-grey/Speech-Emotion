# Speech-Emotion
Demo Link- https://drive.google.com/drive/folders/1P7UaEfS9dKylwA9Li4ORG7fT1qG7JPoz?usp=sharing


🎙️ Emotion Recognition from Speech using ML/DL
This project leverages machine learning and deep learning techniques to classify human emotions from voice recordings. Built on acoustic feature extraction methods, the model runs live predictions through an intuitive Streamlit web application, and was trained using the RAVDESS dataset.

🔍 Project Summary
The system identifies emotional states from .wav audio samples by extracting relevant audio features like MFCCs, Mel-spectrograms, Chroma vectors, and more. A deep learning model is trained on these features and deployed via an interactive web interface for real-time use.

😊 Emotion Categories
Neutral

Calm

Happy

Sad

Angry

Fearful

Surprised

❗ Surprised was excluded from training due to poor model performance in generalizing this class.

🎧 Dataset Details
Dataset: RAVDESS

Comprises speech and song recordings by 24 professional actors

File naming convention: 03-01-02-01-01-01-01.wav (encodes metadata like emotion, intensity, actor, etc.)

The repository includes only processed .npy files; raw .wav files must be downloaded separately and placed in the data directory.

⚙️ Workflow
Load Data: Uses pre-saved .npy feature and label arrays

Preprocessing: Converts raw audio into numerical feature representations

Feature Extraction using:

40 MFCCs

125 Mel-Spectrogram bins

Zero Crossing Rate

Spectral Bandwidth

10 Chroma features

Model Training:

Final model: Deep Neural Network with 179 input features

Deployment:

Real-time inference using a Streamlit-based frontend

📈 Model Results
Model Version	Accuracy	F1-Score
ANN (after removing Surprised)	~82%	~81%
Final ANN Model	82%	81%
