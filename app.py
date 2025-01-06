import streamlit as st
import pickle
import pandas as pd
from nltk.stem.porter import PorterStemmer
from PIL import Image
import easyocr
import numpy as np
import shap
import matplotlib.pyplot as plt

# Initialize the stemmer
ps = PorterStemmer()

# Initialize EasyOCR Reader
reader = easyocr.Reader(['en'])  # Specify language(s)

# Custom CSS for background color and styling
st.markdown("""
    <style>
    body {
        background-color: #f0f8ff;
        color: #333333;
    }
    .block-container {
        padding: 1rem 2rem;
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Function to preprocess and transform text
def transform_text(text):
    # Convert to lowercase, remove newlines, and strip spaces
    text = text.lower().replace("\n", " ").strip()
    words = text.split()  # Split by spaces
    words = [word for word in words if word.isalnum()]  # Remove non-alphanumeric words
    custom_stopwords = set(["the", "and", "is", "in", "to", "it", "of", "for", "on", "this", "a"])
    words = [word for word in words if word not in custom_stopwords]  # Remove stopwords
    words = [ps.stem(word) for word in words]  # Perform stemming
    return " ".join(words)

# Extract text using EasyOCR
def extract_text_from_image(image):
    # Convert PIL image to a NumPy array
    image_array = np.array(image)
    results = reader.readtext(image_array, detail=0)  # Extract text without bounding boxes
    return " ".join(results)

# Load the TF-IDF vectorizer and classifier model
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError:
    st.error("❌ Model or vectorizer file not found. Please ensure the files are in the correct location.")
    st.stop()

# Initialize SHAP explainer
def predict_fn(texts):
    transformed_texts = tfidf.transform(texts)
    return model.predict_proba(transformed_texts)

explainer = shap.Explainer(predict_fn, tfidf)

# Streamlit App
st.title("📧 Email/SMS Spam Classifier")
st.write("### 🔍 Detect Spam in Text, CSV Files, or Images")

tab1, tab2, tab3 = st.tabs(["📝 Text Input", "📂 CSV File Upload", "🖼️ Image Upload"])

# Tab 1: Text Input
with tab1:
    st.write("### Enter Message")
    input_sms = st.text_area("Type your message below:", placeholder="e.g., Congratulations! You've won a $1,000 gift card.")
    if st.button('Classify Text', key='text'):
        if input_sms.strip() == "":
            st.warning("⚠️ Please enter a message to classify.")
        else:
            with st.spinner("🔄 Processing your message..."):
                transformed_sms = transform_text(input_sms)
                vector_input = tfidf.transform([transformed_sms])
                result = model.predict(vector_input)[0]
                st.success("✅ Not Spam" if result == 0 else "🚨 Spam")
                
                # Add SHAP explanation
                st.write("### SHAP Explanation")
                try:
                    # Generate SHAP values
                    shap_values = explainer([transformed_sms])
                    
                    # Display SHAP contributions
                    st.write("#### Contribution of Words to Prediction")
                    fig = shap.plots.text(shap_values[0], display=False)
                    st.pyplot(fig)

                except Exception as e:
                    st.error(f"❌ Error generating SHAP explanation: {e}")
