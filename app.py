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

# Ensure required NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Import IPython for SHAP visualizations
try:
    import IPython
except ImportError:
    st.error("IPython is required for SHAP visualizations. Install it using 'pip install ipython'.")
    st.stop()

# Initialize components
ps = PorterStemmer()
reader = easyocr.Reader(['en'])

# Custom stop words
custom_stopwords = {"the", "and", "is", "in", "to", "it", "of", "for", "on", "this", "a"}

# Text preprocessing function
def transform_text(text):
    text = text.lower().strip()
    words = [word for word in text.split() if word.isalnum()]
    words = [word for word in words if word not in custom_stopwords]
    return " ".join(ps.stem(word) for word in words)

# EasyOCR text extraction function
def extract_text_from_image(image):
    return " ".join(reader.readtext(np.array(image), detail=0))

# Load the TF-IDF vectorizer and classifier model
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model or vectorizer file not found. Ensure 'vectorizer.pkl' and 'model.pkl' are in the working directory.")
    st.stop()

# SHAP explainer
def predict_fn(texts):
    return model.predict_proba(tfidf.transform(texts))

explainer = shap.Explainer(predict_fn, shap.maskers.Text())

# Streamlit app
st.title("📧 Email/SMS Spam Classifier")
tabs = st.tabs(["📝 Text Input", "📂 CSV Upload", "🖼️ Image Upload"])

# Tab 1: Text Input
with tabs[0]:
    input_sms = st.text_area("Enter a message:")
    if st.button("Classify Text"):
        if input_sms.strip():
            transformed_sms = transform_text(input_sms)
            result = model.predict(tfidf.transform([transformed_sms]))[0]
            st.success("✅ Not Spam" if result == 0 else "🚨 Spam")
        else:
            st.warning("Please enter a message.")

    if st.checkbox("Show SHAP Explanation"):
        try:
            shap_values = explainer([input_sms])
            st.write("### SHAP Explanation")
            shap.plots.text(shap_values[0])
        except Exception as e:
            st.error(f"SHAP Error: {e}")

# Tab 2: CSV Upload
with tabs[1]:
    uploaded_file = st.file_uploader("Upload a CSV with a 'message' column", type="csv")
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        if 'message' not in data.columns:
            st.warning("The uploaded file must contain a 'message' column.")
        else:
            data['transformed'] = data['message'].fillna("").apply(transform_text)
            data['classification'] = model.predict(tfidf.transform(data['transformed']))
            data['classification'] = data['classification'].map({1: "Spam", 0: "Not Spam"})
            st.write(data[['message', 'classification']])
            csv = data[['message', 'classification']].to_csv(index=False)
            st.download_button("Download Results", csv, "classified_messages.csv")

# Tab 3: Image Upload
with tabs[2]:
    uploaded_images = st.file_uploader("Upload Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if uploaded_images:
        for img_file in uploaded_images:
            image = Image.open(img_file)
            st.image(image, caption=img_file.name)
            extracted_text = extract_text_from_image(image)
            if extracted_text.strip():
                transformed_text = transform_text(extracted_text)
                prediction = model.predict(tfidf.transform([transformed_text]))[0]
                st.write(f"Classification: {'✅ Not Spam' if prediction == 0 else '🚨 Spam'}")
            else:
                st.warning("No text detected in the image.")
