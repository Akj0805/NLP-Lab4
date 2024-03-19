import streamlit as st
import joblib
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Load the pretrained model
with open('C:/Users/Akshay/Desktop/Deployment/model.pkl', 'rb') as f:
    model = joblib.load(f)

# Load the vectorizer
with open('C:/Users/Akshay/Desktop/Deployment/vectorizer.pkl', 'rb') as f:
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
