import streamlit as st
import joblib
import pickle
import requests
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Function to download file from GitHub
def download_file_from_github(url, destination):
    response = requests.get(url)
    with open(destination, 'wb') as f:
        f.write(response.content)

# GitHub URLs for model and vectorizer
model_url = 'https://github.com/Akj0805/NLP-Lab4/blob/main/model.pkl'
vectorizer_url = 'https://github.com/Akj0805/NLP-Lab4/blob/main/vectorizer.pkl'

# Download model and vectorizer files
download_file_from_github(model_url, 'model.pkl')
download_file_from_github(vectorizer_url, 'vectorizer.pkl')

# Load the pretrained model
with open('model.pkl', 'rb') as f:
    model = joblib.load(f)

# Load the vectorizer
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = joblib.load(f)

# Initialize NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Define stop words and initialize lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Define function to preprocess text
def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text)
    # Stop word removal and lowercasing
    filtered_tokens = [word.lower() for word in tokens if word.lower() not in stop_words and word.isalnum()]
    # Lemmatization
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    return ' '.join(lemmatized_tokens)

# Define the UI
st.title('Yelp Review Sentiment Analysis')

# Define input fields
review_text = st.text_area("Enter your review here:")

# Define prediction logic
if st.button('Predict Sentiment'):
    if review_text.strip() == '':
        st.error("Please enter some review text.")
    else:
        # Preprocess the input text
        preprocessed_text = preprocess_text(review_text)
        # Vectorize the preprocessed text
        input_vectorized = vectorizer.transform([preprocessed_text])
        # Make prediction using the model
        prediction = model.predict(input_vectorized)
        # Display prediction
        if prediction[0] == 1:
            st.success("Positive Sentiment")
        else:
            st.error("Negative Sentiment")
