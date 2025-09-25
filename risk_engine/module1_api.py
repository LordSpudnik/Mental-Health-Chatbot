# import joblib
# from flask import Flask, request, jsonify
# import re
# import nltk
# from transformers import pipeline
# import warnings

# # --- Setup and Model Loading ---

# # Suppress warnings for a cleaner output
# warnings.filterwarnings("ignore", category=UserWarning, module='transformers')

# # Initialize Flask app
# app = Flask(__name__)

# # --- Load all models and tools ONCE at startup ---
# print("--- Loading all models. This may take a moment. ---")

# # Load the custom-trained vectorizer and risk classifier
# try:
#     vectorizer_path = '../scripts/tfidf_vectorizer.pkl'
#     model_path = '../scripts/risk_classifier_model.pkl'
    
#     vectorizer = joblib.load(vectorizer_path)
#     model = joblib.load(model_path)
#     print("Custom risk classifier and vectorizer loaded successfully.")
# except FileNotFoundError:
#     print("FATAL ERROR: Model files (tfidf_vectorizer.pkl, risk_classifier_model.pkl) not found.")
#     print("Please run final_analyzer.py first to train and save these files.")
#     exit()

# # Load the pre-trained general sentiment pipeline from Hugging Face
# try:
#     sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
#     print("General sentiment model loaded successfully.")
# except Exception as e:
#     print(f"FATAL ERROR: Could not load sentiment pipeline. Error: {e}")
#     exit()

# # Load NLTK data and define the preprocessing function
# try:
#     nltk.data.find('corpora/stopwords')
#     nltk.data.find('corpora/wordnet')
# except LookupError:
#     nltk.download('stopwords')
#     nltk.download('wordnet')

# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer

# def preprocess_text(text):
#     """Preprocessing function - MUST be identical to the one used for training."""
#     if not isinstance(text, str):
#         return ""
#     lemmatizer = WordNetLemmatizer()
#     stop_words = set(stopwords.words('english'))
#     text = text.lower()
#     text = re.sub(r"[^a-zA-Z\s']", '', text)
#     tokens = text.split()
#     processed_tokens = [
#         lemmatizer.lemmatize(word) for word in tokens if word not in stop_words
#     ]
#     return ' '.join(processed_tokens)

# print("--- All models loaded. API is ready to accept requests. ---")

# # --- API Endpoint ---

# @app.route('/analyze', methods=['POST'])
# def analyze_sentiment():
#     """
#     Main API endpoint to perform sentiment analysis.
#     Expects JSON input: {"text": "Some user message here."}
#     """
#     # Get the JSON data from the request
#     data = request.get_json()
#     if not data or 'text' not in data:
#         return jsonify({"error": "Invalid input. Please provide JSON with a 'text' key."}), 400

#     text_input = data['text']
#     if not isinstance(text_input, str) or not text_input.strip():
#         return jsonify({"error": "Text input cannot be empty."}), 400

#     # 1. Get general sentiment (positive, negative, neutral) and confidence
#     sentiment_result = sentiment_pipeline(text_input)[0]
#     sentiment_label = sentiment_result['label'].lower()
#     confidence = round(sentiment_result['score'], 2)

#     # 2. Preprocess the text for the custom model
#     processed_input = preprocess_text(text_input)
#     vectorized_input = vectorizer.transform([processed_input])
    
#     # 3. Get custom risk classification (high-risk, low-risk)
#     custom_prediction = custom_model.predict(vectorized_input)[0]
#     tone = 'high-risk' if custom_prediction == 'suicide' else 'low-risk'

#     # 4. Assemble the final JSON output
#     final_output = {
#         "text": text_input,
#         "sentiment": sentiment_label,
#         "tone": tone,
#         "confidence_score": confidence
#     }

#     # Return the result as a JSON response
#     return jsonify(final_output)

# # --- Main Execution Block ---

# if __name__ == '_main_':
#     # Run the Flask server
#     # Using port 5001 for Module 1 to avoid conflicts with other modules
#     app.run(debug=True, port=5001)

import joblib
from flask import Flask, request, jsonify
import re
import nltk
from transformers import pipeline
import warnings
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- Setup and Model Loading ---

warnings.filterwarnings("ignore", category=UserWarning, module='transformers')
app = Flask(__name__)

# --- Load all models and tools ONCE at startup ---
print("--- MODULE 1: Loading all models. This may take a moment. ---")

try:
    vectorizer = joblib.load('../scripts/tfidf_vectorizer.pkl')
    custom_model = joblib.load('../scripts/risk_classifier_model.pkl')
    print("Custom risk classifier and vectorizer loaded successfully.")
    
    sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
    print("General sentiment model loaded successfully.")
except FileNotFoundError:
    print("FATAL ERROR: Model files not found. Please ensure they are in the 'scripts' folder.")
    exit()
except Exception as e:
    print(f"FATAL ERROR: Could not load models. Error: {e}")
    exit()

# --- Text Preprocessing Function (Must match the training script) ---
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s']", '', text)
    tokens = text.split()
    processed_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(processed_tokens)

# --- API Endpoint ---
@app.route('/analyze', methods=['POST'])
def analyze_text():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Invalid input. Please provide JSON with a 'text' key."}), 400

    text_input = data['text']
    
    # 1. Get sentiment
    sentiment_result = sentiment_pipeline(text_input)[0]
    sentiment_label = sentiment_result['label'].lower()
    confidence = round(sentiment_result['score'], 2)

    # 2. Get custom risk classification
    processed_input = preprocess_text(text_input)
    vectorized_input = vectorizer.transform([processed_input])
    custom_prediction = custom_model.predict(vectorized_input)[0]
    tone = 'high-risk' if custom_prediction == 'suicide' else 'low-risk'

    final_output = {
        "text": text_input,
        "sentiment": sentiment_label,
        "tone": tone,
        "confidence_score": confidence
    }
    return jsonify(final_output)

# --- Main Execution ---
if __name__ == '__main__':
    # Ensure NLTK data is downloaded
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('stopwords')
        nltk.download('wordnet')
    
    # Run the server for Module 1 on port 5001
    app.run(debug=True, port=5001)