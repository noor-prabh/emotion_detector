import streamlit as st
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import joblib

model = load_model('emotion_cnn_model.h5')
scaler = joblib.load('emotion_scaler.pkl')
le = joblib.load('emotion_label_encoder.pkl')


def extract_features(file_path):
    signal, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)
    mfcc_scaled = np.mean(mfcc.T, axis=0)
    return mfcc_scaled

def predict_emotion(audio_file):
    features = extract_features(audio_file)
    features = features.reshape(40,1)
    features = np.expand_dims(features, axis=0)
    prediction = model.predict(features)[0]
    emotion_index = np.argmax(prediction)
    emotion = le.inverse_transform([emotion_index])[0]
    return emotion

st.title("🎙️ Speech Emotion Recognition")
st.write("Upload a `.wav` file to predict the emotion.")

uploaded_file = st.file_uploader("Choose a .wav file", type="wav")

if uploaded_file is not None:
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())
    st.audio(uploaded_file, format='audio/wav')
    emotion = predict_emotion("temp.wav")
    st.success(f"Predicted Emotion: **{emotion}**")