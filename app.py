import streamlit as st
import joblib
import pandas as pd
import re
import numpy as np

# Load the pre-trained model and vectorizer
try:
    model = joblib.load('./xgboost_model_compressed (1).joblib')
    vectorizer = joblib.load('./tfidf_vectorizer.pkl')
except Exception as e:
    st.error(f"Error loading model or vectorizer: {e}")
    st.stop()

# Function to clean text
def clean_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

# Streamlit app
st.title('Sentiment Analysis App')

# Input text from the user
user_input = st.text_area("Enter text for sentiment analysis:")

if st.button('Predict Sentiment'):
    if user_input:
        try:
            # Clean and preprocess the input text
            cleaned_text = clean_text(user_input)
            
            # Transform the text to feature vector
            text_vector = vectorizer.transform([cleaned_text])
            
            # Check the shape of the text_vector
            st.write(f"Feature vector shape: {text_vector.shape}")
            
            # Predict sentiment
            prediction = model.predict(text_vector)
            sentiment = ['Negative', 'Neutral', 'Positive'][prediction[0]]
            
            st.write(f"The sentiment of the input text is: **{sentiment}**")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        st.write("Please enter some text.")
