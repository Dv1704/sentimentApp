# Sentiment Analysis App

This is a simple Streamlit app for performing sentiment analysis on user-provided text using a pre-trained XGBoost model and a TF-IDF vectorizer. The app cleans the input text and predicts whether the sentiment is **Positive**, **Neutral**, or **Negative**.

## Features

- Input text for sentiment analysis.
- Text is cleaned by removing URLs, punctuation, numbers, and extra spaces.
- The app uses a TF-IDF vectorizer to convert text into feature vectors.
- The sentiment is predicted using a pre-trained XGBoost model.
- Sentiment results are displayed as **Positive**, **Neutral**, or **Negative**.

## Requirements

To run this app, you'll need Python 3.8 or later and the following packages:

- `streamlit==1.38.0`
- `joblib==1.4.2`
- `pandas==2.2.2`
- `scikit-learn==1.3.2`
- `numpy==1.26.4`
- `xgboost==1.6.2`

