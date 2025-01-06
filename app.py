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
import os

# Set the NLTK data path to a custom directory within your project
nltk.data.path.append(os.path.join(os.getcwd(), 'nltk_data'))

# Download necessary NLTK resources to this custom directory if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Punkt tokenizer not found. Downloading...")
    nltk.download('punkt', download_dir=os.path.join(os.getcwd(), 'nltk_data'))

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Stopwords not found. Downloading...")
    nltk.download('stopwords', download_dir=os.path.join(os.getcwd(), 'nltk_data'))

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
                shap_values = explainer([input_sms])
                tokens = shap_values.data[0]  # Extract tokens
                contributions = shap_values.values[0]  # SHAP contributions
                
                # Create a DataFrame for better organization
                shap_df = pd.DataFrame({
                    'Token': tokens,
                    'Contribution': contributions
                })
                shap_df['Direction'] = shap_df['Contribution'].apply(lambda x: "Spam" if x > 0 else "Not Spam")
                shap_df = shap_df.sort_values('Contribution', key=abs, ascending=False).head(10)  # Top 10 contributors
                
                # Display bar chart for SHAP contributions
                st.write("#### Top Words Contributing to the Prediction")
                fig, ax = plt.subplots()
                shap_df.plot.barh(x='Token', y='Contribution', 
                                  color=shap_df['Direction'].map({"Spam": "red", "Not Spam": "green"}), ax=ax)
                plt.title("SHAP Contributions")
                plt.xlabel("SHAP Value")
                plt.ylabel("Token")
                st.pyplot(fig)
                
                # Detailed explanation in text
                st.write("#### Explanation Summary")
                for _, row in shap_df.iterrows():
                    token, contribution, direction = row['Token'], row['Contribution'], row['Direction']
                    st.markdown(f"- *{token}*: {direction} ({'+' if contribution > 0 else ''}{contribution:.2f})")
                
                # Interactive SHAP HTML visualization
                st.write("#### Detailed SHAP Text Explanation")
                shap_html = shap.plots.text(shap_values[0], display=False)  # Generate SHAP text as HTML
                st.components.v1.html(shap_html, height=400)  # Embed the HTML in Streamlit
                    
            except Exception as e:
                st.error(f"❌ Error generating SHAP explanation: {e}")

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
                    # Preprocess and classify
                    data['transformed_message'] = data['message'].apply(transform_text)
                    vector_input = tfidf.transform(data['transformed_message'])
                    predictions = model.predict(vector_input)  # Returns a NumPy array
                    
                    # Convert predictions to a Pandas Series and map values
                    data['classification'] = pd.Series(predictions).map({1: "Spam", 0: "Not Spam"})
                    
                    # Display results
                    st.write(data[['message', 'classification']])
                    
                    # Allow CSV download
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
