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

# Download NLTK data files
nltk.download('stopwords')
nltk.download('punkt')

# Initialize the stemmer
ps = PorterStemmer()

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
                # Preprocess and classify
                transformed_sms = transform_text(input_sms)
                vector_input = tfidf.transform([transformed_sms])
                result = model.predict(vector_input)[0]
                
                # Display classification result
                st.success("✅ Not Spam" if result == 0 else "🚨 Spam")
                
    # SHAP Explanation Option (for text only)
    if st.checkbox("Show Explanation", key='shap_checkbox'):
        if not input_sms.strip():
            st.warning("⚠️ Please enter a message to display the explanation.")
        else:
            st.write("### SHAP Explanation")
            try:
                # Generate SHAP explanation
                vector_input = tfidf.transform([transformed_sms])
                shap_values = explainer(vector_input)

                # Display SHAP contributions
                st.write("#### Contribution of Words to Prediction")
                fig, ax = plt.subplots(figsize=(10, 5))
                shap.summary_plot(shap_values, vector_input, feature_names=tfidf.get_feature_names_out(), plot_type="bar", show=False)
                st.pyplot(fig)

            except Exception as e:
                st.error(f"❌ Error generating SHAP explanation: {e}")
