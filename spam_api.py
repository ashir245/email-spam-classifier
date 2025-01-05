from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
import pickle
import pandas as pd
import pytesseract
from PIL import Image
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
import logging
from PIL import ImageEnhance

# Initialize FastAPI
app = FastAPI(
    title="Spam Classifier API",
    description="An API for classifying text, CSV files, and images for spam detection.",
    version="1.0"
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Download NLTK stopwords
nltk.download('stopwords')
nltk.download('punkt')
ps = PorterStemmer()

# Load Model and Vectorizer
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
    logger.info("Model and vectorizer loaded successfully.")
except FileNotFoundError:
    logger.error("Model or vectorizer file not found.")
    raise RuntimeError("Model or vectorizer file not found. Please upload the required files.")

# Text transformation function
def transform_text(text):
    text = text.lower()  # Convert to lowercase
    text = text.replace("\n", " ").strip()  # Normalize whitespace
    text = nltk.word_tokenize(text)  # Tokenize
    text = [i for i in text if i.isalnum()]  # Remove non-alphanumeric characters
    text = [i for i in text if i not in stopwords.words('english') and i not in string.punctuation]  # Remove stopwords
    text = [ps.stem(i) for i in text]  # Apply stemming
    return " ".join(text)

# Request model for text classification
class TextRequest(BaseModel):
    message: str

@app.post("/classify_text")
def classify_text(request: TextRequest):
    """
    Classifies a single text message as Spam or Not Spam.
    """
    try:
        transformed_text = transform_text(request.message)
        vector_input = tfidf.transform([transformed_text])
        prediction = model.predict(vector_input)[0]
        classification = "Spam" if prediction == 1 else "Not Spam"
        return {"message": request.message, "classification": classification}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

@app.post("/classify_csvs")
def classify_csvs(files: List[UploadFile] = File(...)):
    """
    Classifies messages from multiple CSV files.
    """
    results = []
    for file in files:
        try:
            data = pd.read_csv(file.file)
            if 'message' not in data.columns:
                results.append({"file": file.filename, "error": "The uploaded file must have a 'message' column."})
                continue
            
            data['transformed_message'] = data['message'].apply(transform_text)
            vector_input = tfidf.transform(data['transformed_message'])
            data['classification'] = model.predict(vector_input)
            data['classification'] = data['classification'].apply(lambda x: "Spam" if x == 1 else "Not Spam")
            
            results.append({
                "file": file.filename,
                "classifications": data[['message', 'classification']].to_dict(orient="records")
            })
        except pd.errors.EmptyDataError:
            results.append({"file": file.filename, "error": "The uploaded file is empty."})
        except Exception as e:
            results.append({"file": file.filename, "error": f"Error processing file: {str(e)}"})
    return results

@app.post("/classify_images")
def classify_images(files: List[UploadFile] = File(...)):
    """
    Classifies text extracted from multiple uploaded images.
    """
    results = []
    for file in files:
        try:
            # Open the image file
            image = Image.open(file.file)
            
            # Preprocess the image (grayscale and contrast enhancement)
            gray_image = image.convert('L')  # Convert to grayscale
            enhancer = ImageEnhance.Contrast(gray_image)
            enhanced_image = enhancer.enhance(2)  # Enhance contrast
            
            # Extract text using Tesseract OCR
            extracted_text = pytesseract.image_to_string(enhanced_image, config="--psm 6 --oem 3").strip()
            
            # Check if any text was extracted
            if not extracted_text:
                results.append({"file": file.filename, "classification": "No meaningful text found"})
                continue
            
            # Preprocess and classify the extracted text
            transformed_text = transform_text(extracted_text)
            vector_input = tfidf.transform([transformed_text])
            prediction = model.predict(vector_input)[0]
            classification = "Spam" if prediction == 1 else "Not Spam"
            
            # Add result for the current file
            results.append({"file": file.filename, "classification": classification})
        except Exception as e:
            # Handle errors for individual files
            results.append({"file": file.filename, "error": f"Error processing file: {str(e)}"})
    return results

@app.get("/")
def get_debug_info():
    """
    Provides a description of the API and its endpoints.
    """
    return {
        "message": "Welcome to the Spam Classifier API!",
        "endpoints": {
            "/classify_text": {
                "method": "POST",
                "description": "Classify a single text message as Spam or Not Spam."
            },
            "/classify_csvs": {
                "method": "POST",
                "description": "Upload multiple CSV files with a 'message' column to classify each message."
            },
            "/classify_images": {
                "method": "POST",
                "description": "Upload multiple images to classify text extracted from each image."
            }
        }
    }
