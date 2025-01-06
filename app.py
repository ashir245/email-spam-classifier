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
    return model.predict(transformed_texts)

explainer = shap.Explainer(predict_fn, shap.maskers.Text())

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

# Tab 2: CSV File Upload
with tab2:
    st.write("### Upload CSV Files")
    uploaded_files = st.file_uploader("Upload one or more CSV files with a 'message' column.", type=["csv"], accept_multiple_files=True)
    if uploaded_files and st.button('Classify CSVs', key='csv_batch'):
        for uploaded_file in uploaded_files:
            try:
                st.write(f"### Results for `{uploaded_file.name}`")
                data = pd.read_csv(uploaded_file)
                if 'message' not in data.columns:
                    st.warning(f"⚠️ No 'message' column in {uploaded_file.name}.")
                    continue
                with st.spinner(f"🔄 Processing '{uploaded_file.name}'..."):
                    data['transformed_message'] = data['message'].apply(transform_text)
                    vector_input = tfidf.transform(data['transformed_message'])
                    predictions = model.predict(vector_input)
                    data['classification'] = pd.Series(predictions).map({1: "Spam", 0: "Not Spam"})
                    st.write(data[['message', 'classification']])
                    csv = data[['message', 'classification']].to_csv(index=False)
                    st.download_button(
                        label=f"📥 Download Results for {uploaded_file.name}",
                        data=csv,
                        file_name=f"{uploaded_file.name.split('.')[0]}_results.csv",
                        mime='text/csv'
                    )
            except Exception as e:
                st.error(f"❌ Error with file '{uploaded_file.name}': {e}")

# Tab 3: Image Upload
with tab3:
    st.write("### Upload Images")
    uploaded_images = st.file_uploader("Upload images to extract and classify text.", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
    if uploaded_images and st.button('Classify Images', key='image_batch'):
        for image_file in uploaded_images:
            try:
                image = Image.open(image_file)
                st.image(image, caption=image_file.name)
                extracted_text = extract_text_from_image(image)
                if extracted_text.strip():
                    transformed_text = transform_text(extracted_text)
                    vector_input = tfidf.transform([transformed_text])
                    prediction = model.predict(vector_input)[0]
                    st.write(f"Classification: {'✅ Not Spam' if prediction == 0 else '🚨 Spam'}")
                else:
                    st.warning(f"⚠️ No text found in {image_file.name}.")
            except Exception as e:
                st.error(f"❌ Error with image {image_file.name}: {e}")
