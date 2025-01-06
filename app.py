import streamlit as st
import pickle
import string
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from PIL import Image
import pytesseract
import shap
import matplotlib.pyplot as plt

# Check and download NLTK data
for resource in ['stopwords', 'punkt']:
    try:
        nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')
    except LookupError:
        nltk.download(resource)

# Initialize the stemmer
ps = PorterStemmer()

# Function to preprocess and transform text
def transform_text(text):
    text = text.lower().replace("\n", " ").strip()
    text = nltk.word_tokenize(text)
    text = [word for word in text if word.isalnum()]
    text = [word for word in text if word not in stopwords.words('english')]
    text = [ps.stem(word) for word in text]
    return " ".join(text)

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
    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                data = pd.read_csv(uploaded_file)
                if 'message' not in data.columns:
                    st.warning(f"⚠️ No 'message' column in {uploaded_file.name}.")
                    continue
                data['transformed_message'] = data['message'].apply(transform_text)
                vector_input = tfidf.transform(data['transformed_message'])
                predictions = model.predict(vector_input)
                data['classification'] = ["Spam" if pred == 1 else "Not Spam" for pred in predictions]
                st.write(data[['message', 'classification']])
            except Exception as e:
                st.error(f"❌ Error with file '{uploaded_file.name}': {e}")

# Tab 3: Image Upload
with tab3:
    st.write("### Upload Images")
    uploaded_images = st.file_uploader("Upload images to extract and classify text.", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
    if uploaded_images:
        for image_file in uploaded_images:
            try:
                image = Image.open(image_file)
                st.image(image, caption=image_file.name)
                extracted_text = pytesseract.image_to_string(image)
                if extracted_text.strip():
                    transformed_text = transform_text(extracted_text)
                    vector_input = tfidf.transform([transformed_text])
                    prediction = model.predict(vector_input)[0]
                    st.write(f"Classification: {'✅ Not Spam' if prediction == 0 else '🚨 Spam'}")
                else:
                    st.warning(f"⚠️ No text found in {image_file.name}.")
            except Exception as e:
                st.error(f"❌ Error with image {image_file.name}: {e}")
