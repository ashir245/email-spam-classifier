import streamlit as st
import pickle
import pandas as pd
from nltk.stem.porter import PorterStemmer
from PIL import Image
import easyocr
import numpy as np
import shap
import matplotlib.pyplot as plt
import nltk

# Download NLTK data files
nltk.download('stopwords')
nltk.download('punkt')

# Initialize components
ps = PorterStemmer()
reader = easyocr.Reader(['en'])

# Custom stopwords
custom_stopwords = {"the", "and", "is", "in", "to", "it", "of", "for", "on", "this", "a"}

# Preprocessing Function
def transform_text(text):
    text = text.lower().strip()
    words = [word for word in text.split() if word.isalnum()]
    words = [word for word in words if word not in custom_stopwords]
    return " ".join(ps.stem(word) for word in words)

# EasyOCR Text Extraction
def extract_text_from_image(image):
    return " ".join(reader.readtext(np.array(image), detail=0))

# Load models
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model or vectorizer file not found.")
    st.stop()

# SHAP Explainer
def predict_fn(texts):
    return model.predict_proba(tfidf.transform(texts))

explainer = shap.Explainer(predict_fn, shap.maskers.Text())

# Streamlit App
st.title("📧 Email/SMS Spam Classifier")
tabs = st.tabs(["📝 Text Input", "📂 CSV Upload", "🖼️ Image Upload"])

# Tab 1: Text Input
with tabs[0]:
    input_sms = st.text_area("Enter a message:")
    if st.button("Classify"):
        if input_sms.strip():
            transformed = transform_text(input_sms)
            result = model.predict(tfidf.transform([transformed]))[0]
            st.success("✅ Not Spam" if result == 0 else "🚨 Spam")
        else:
            st.warning("Please enter a message.")

    if st.checkbox("Show Explanation"):
        try:
            shap_values = explainer([input_sms])
            st.write("### SHAP Explanation")
            shap.plots.text(shap_values[0])
        except Exception as e:
            st.error(f"SHAP Error: {e}")

# Tab 2: CSV Upload
with tabs[1]:
    uploaded_file = st.file_uploader("Upload CSV with a 'message' column", type="csv")
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        if 'message' not in data.columns:
            st.warning("The uploaded file must contain a 'message' column.")
        else:
            data['transformed'] = data['message'].fillna("").apply(transform_text)
            data['classification'] = model.predict(tfidf.transform(data['transformed']))
            st.write(data[['message', 'classification']].rename(columns={'classification': 'Spam/Not Spam'}))
            st.download_button(
                "Download Results",
                data.to_csv(index=False),
                file_name="classified_messages.csv"
            )

# Tab 3: Image Upload
with tabs[2]:
    uploaded_images = st.file_uploader("Upload Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if uploaded_images:
        for img_file in uploaded_images:
            image = Image.open(img_file)
            st.image(image, caption=img_file.name)
            extracted_text = extract_text_from_image(image)
            if extracted_text.strip():
                transformed = transform_text(extracted_text)
                prediction = model.predict(tfidf.transform([transformed]))[0]
                st.write(f"Classification: {'✅ Not Spam' if prediction == 0 else '🚨 Spam'}")
            else:
                st.warning("No text detected in the image.")
