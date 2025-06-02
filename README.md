# Emotion Recognition from Speech

This project aims to recognize human emotions from speech audio using deep learning and signal processing techniques. It uses MFCC features and a neural network model to classify emotions into eight categories.

## Project Overview

- **Objective**: Detect emotions such as happy, sad, angry, etc., from `.wav` audio files.
- **Model Accuracy**: Approximately 71% across 8 emotion classes.
- **Dataset Used**: [RAVDESS](https://zenodo.org/record/1188976) - a well-known emotional speech dataset.
- **Techniques**: MFCC feature extraction, label encoding, and neural network-based classification.

## Key Features

- Extracts MFCC features from audio files
- Encodes categorical emotion labels
- Trains and evaluates a neural network model
- Streamlit-based web interface for real-time predictions

## Getting Started

### Install Dependencies

```bash
pip install -r requirements.txt
