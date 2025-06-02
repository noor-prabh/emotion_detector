import os
import librosa
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


data_path = "D:\machine learning\emotion_detection\data"
emotion_map = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

features = []
for actor_folder in os.listdir(data_path):
    actor_path = os.path.join(data_path, actor_folder)
    if not os.path.isdir(actor_path):
        continue

    for file in os.listdir(actor_path):
        if not file.endswith(".wav"):
            continue
        try:
            parts = file.split("-")
            emotion_code = parts[2]
            emotion = emotion_map.get(emotion_code)
            if emotion is None:
                continue

            file_path = os.path.join(actor_path, file) 
            signal, sr = librosa.load(file_path, sr=None)
            mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)
            mfccs_scaled = np.mean(mfccs.T, axis=0)

            features.append({"feature":mfccs_scaled, "emotion":emotion})

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

df = pd.DataFrame(features)
df.to_pickle("features.pkl")
print("feature extraction complete")

