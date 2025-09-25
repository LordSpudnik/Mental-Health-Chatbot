# # import joblib
# # from flask import Flask, request, jsonify
# # import numpy as np
# # import re
# # import nltk
# # from nltk.corpus import stopwords
# # from nltk.stem import WordNetLemmatizer

# # # --- Setup and Model Loading ---

# # # Initialize Flask app
# # app = Flask(__name__)

# # # Load the pre-trained TF-IDF vectorizer and the logistic regression model from Module 1
# # print("--- Loading models from Module 1 ---")
# # try:
# #     # Go up one level from 'risk_engine' and then into 'scripts'
# #     vectorizer_path = '../scripts/tfidf_vectorizer.pkl'
# #     model_path = '../scripts/risk_classifier_model.pkl'
    
# #     vectorizer = joblib.load(vectorizer_path)
# #     model = joblib.load(model_path)
# #     print("Models loaded successfully.")
# # except FileNotFoundError:
# #     print("Error: Model or vectorizer files not found.")
# #     print("Please run the updated Module 1 script first to generate them.")
# #     exit()

# # # Text preprocessing function (must be identical to the one used for training)
# # def preprocess_text(text):
# #     if not isinstance(text, str):
# #         return ""
# #     lemmatizer = WordNetLemmatizer()
# #     stop_words = set(stopwords.words('english'))
# #     text = text.lower()
# #     text = re.sub(r"[^a-zA-Z\s']", '', text)
# #     tokens = text.split()
# #     processed_tokens = [
# #         lemmatizer.lemmatize(word) for word in tokens if word not in stop_words
# #     ]
# #     return ' '.join(processed_tokens)

# # # --- Scoring Logic ---

# # def score_phq9(answers):
# #     """Calculates and interprets the PHQ-9 depression score."""
# #     if len(answers) != 9:
# #         return {"error": "PHQ-9 requires exactly 9 answers."}
    
# #     score = sum(answers)
    
# #     if 0 <= score <= 4:
# #         level = "minimal depression"
# #     elif 5 <= score <= 9:
# #         level = "mild depression"
# #     elif 10 <= score <= 14:
# #         level = "moderate depression"
# #     elif 15 <= score <= 19:
# #         level = "moderately severe depression"
# #     else: # 20 <= score <= 27
# #         level = "severe depression"
        
# #     return {"questionnaire": "PHQ-9", "risk_score": score, "level": level}

# # def score_gad7(answers):
# #     """Calculates and interprets the GAD-7 anxiety score."""
# #     if len(answers) != 7:
# #         return {"error": "GAD-7 requires exactly 7 answers."}
        
# #     score = sum(answers)
    
# #     if 0 <= score <= 4:
# #         level = "minimal anxiety"
# #     elif 5 <= score <= 9:
# #         level = "mild anxiety"
# #     elif 10 <= score <= 14:
# #         level = "moderate anxiety"
# #     else: # 15 <= score <= 21
# #         level = "severe anxiety"
        
# #     return {"questionnaire": "GAD-7", "risk_score": score, "level": level}


# # # --- API Endpoints ---

# # @app.route('/score/questionnaire', methods=['POST'])
# # def handle_questionnaire():
# #     """
# #     API endpoint to score a questionnaire (PHQ-9 or GAD-7).
# #     Expects JSON: {"type": "phq9" or "gad7", "answers": [0, 1, 2, ...]}
# #     """
# #     data = request.get_json()
# #     q_type = data.get('type', '').lower()
# #     answers = data.get('answers', [])

# #     if not all(isinstance(i, int) for i in answers):
# #         return jsonify({"error": "Answers must be a list of integers."}), 400

# #     if q_type == 'phq9':
# #         result = score_phq9(answers)
# #     elif q_type == 'gad7':
# #         result = score_gad7(answers)
# #     else:
# #         return jsonify({"error": "Invalid questionnaire type. Use 'phq9' or 'gad7'."}), 400

# #     if "error" in result:
# #         return jsonify(result), 400
        
# #     return jsonify(result)

# # @app.route('/score/text', methods=['POST'])
# # def handle_text_scoring():
# #     """
# #     API endpoint to score risk based on conversation text using Module 1's model.
# #     Expects JSON: {"text": "Some user message here."}
# #     """
# #     data = request.get_json()
# #     text_input = data.get('text', '')

# #     if not text_input:
# #         return jsonify({"error": "Text input cannot be empty."}), 400

# #     # Preprocess and predict
# #     processed_input = preprocess_text(text_input)
# #     vectorized_input = vectorizer.transform([processed_input])
    
# #     # Get the predicted class and the corresponding probability
# #     prediction = model.predict(vectorized_input)[0]
# #     probability = np.max(model.predict_proba(vectorized_input)) # This gets the probability of the predicted class

# #     # --- DEBUGGING LINE ---
# #     # This will print the raw probabilities in your terminal for us to see.
# #     # The order of classes is shown in model.classes_
# #     print(f"DEBUG INFO -> Text: '{text_input}' | Prediction: {prediction} | Confidence: {probability:.2f} | All Probs: {model.predict_proba(vectorized_input)} | Classes: {model.classes_}")

# #     return jsonify({
# #         "text": text_input,
# #         "risk_classification": prediction,
# #         "confidence": round(probability, 2)
# #     })
# # # --- Main Execution ---

# # if __name__ == '__main__':
# #     # Download necessary NLTK data if not present
# #     try:
# #         nltk.data.find('corpora/stopwords')
# #     except LookupError:
# #         nltk.download('stopwords')
# #     try:
# #         nltk.data.find('corpora/wordnet')
# #     except LookupError:
# #         nltk.download('wordnet')
    
# #     # Run the Flask server
# #     app.run(debug=True, port=5002) # Using port 5002 for Module 2



# import joblib
# from flask import Flask, request, jsonify
# import numpy as np
# import re
# import nltk
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer

# # --- Setup and Model Loading ---
# app = Flask(__name__)

# # Load the pre-trained TF-IDF vectorizer and the logistic regression model
# print("--- Loading trained models ---")
# try:
#     # These files should be in the same 'scripts' folder
#     vectorizer_path = '../scripts/tfidf_vectorizer.pkl'
#     model_path = '../scripts/risk_classifier_model.pkl'
    
#     vectorizer = joblib.load(vectorizer_path)
#     model = joblib.load(model_path)
#     print("Models loaded successfully.")
# except FileNotFoundError:
#     print("Error: Model files not found. Please run the updated final_analyzer.py first to generate them.")
#     exit()

# # Text preprocessing function (must be identical to the one used for training)
# def preprocess_text(text):
#     if not isinstance(text, str):
#         return ""
#     lemmatizer = WordNetLemmatizer()
#     stop_words = set(stopwords.words('english'))
#     text = text.lower()
#     text = re.sub(r"[^a-zA-Z\s']", '', text)
#     tokens = text.split()
#     processed_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
#     return ' '.join(processed_tokens)

# # --- Scoring Logic ---
# def score_phq9(answers):
#     score = sum(answers)
#     if 0 <= score <= 4: level = "minimal depression"
#     elif 5 <= score <= 9: level = "mild depression"
#     elif 10 <= score <= 14: level = "moderate depression"
#     elif 15 <= score <= 19: level = "moderately severe depression"
#     else: level = "severe depression"
#     return {"questionnaire": "PHQ-9", "risk_score": score, "level": level}

# def score_gad7(answers):
#     score = sum(answers)
#     if 0 <= score <= 4: level = "minimal anxiety"
#     elif 5 <= score <= 9: level = "mild anxiety"
#     elif 10 <= score <= 14: level = "moderate anxiety"
#     else: level = "severe anxiety"
#     return {"questionnaire": "GAD-7", "risk_score": score, "level": level}

# # --- API Endpoints ---
# @app.route('/score/questionnaire', methods=['POST'])
# def handle_questionnaire():
#     data = request.get_json()
#     q_type = data.get('type', '').lower()
#     answers = data.get('answers', [])
#     if q_type == 'phq9' and len(answers) == 9: result = score_phq9(answers)
#     elif q_type == 'gad7' and len(answers) == 7: result = score_gad7(answers)
#     else: return jsonify({"error": "Invalid input."}), 400
#     return jsonify(result)

# @app.route('/score/text', methods=['POST'])
# def handle_text_scoring():
#     data = request.get_json()
#     text_input = data.get('text', '')
#     if not text_input:
#         return jsonify({"error": "Text input cannot be empty."}), 400

#     processed_input = preprocess_text(text_input)
#     vectorized_input = vectorizer.transform([processed_input])
    
#     prediction = model.predict(vectorized_input)[0]
#     probabilities = model.predict_proba(vectorized_input)
#     confidence = np.max(probabilities)

#     print(f"DEBUG -> Text: '{text_input}' | Prediction: {prediction} | Confidence: {confidence:.2f}")

#     return jsonify({
#         "text": text_input,
#         "risk_classification": prediction,
#         "confidence": round(confidence, 2)
#     })

# # --- Main Execution ---
# if __name__== '_main_':
#     try:
#         nltk.data.find('corpora/stopwords')
#         nltk.data.find('corpora/wordnet')
#     except LookupError:
#         nltk.download('stopwords')
#         nltk.download('wordnet')
    
#     app.run(debug=True, port=5002)


from flask import Flask, request, jsonify

# --- Setup ---
app = Flask(__name__)
print("--- MODULE 2: Ready for questionnaire scoring. ---")

# --- Scoring Logic ---

def score_phq9(answers):
    """Calculates and interprets the PHQ-9 depression score."""
    score = sum(answers)
    level = "minimal depression"
    if 5 <= score <= 9: level = "mild depression"
    elif 10 <= score <= 14: level = "moderate depression"
    elif 15 <= score <= 19: level = "moderately severe depression"
    elif score >= 20: level = "severe depression"
    return {"questionnaire": "PHQ-9", "risk_score": score, "level": level}

def score_gad7(answers):
    """Calculates and interprets the GAD-7 anxiety score."""
    score = sum(answers)
    level = "minimal anxiety"
    if 5 <= score <= 9: level = "mild anxiety"
    elif 10 <= score <= 14: level = "moderate anxiety"
    elif score >= 15: level = "severe anxiety"
    return {"questionnaire": "GAD-7", "risk_score": score, "level": level}

# --- API Endpoint ---
@app.route('/score/questionnaire', methods=['POST'])
def handle_questionnaire():
    data = request.get_json()
    q_type = data.get('type', '').lower()
    answers = data.get('answers', [])

    if not all(isinstance(i, int) for i in answers):
        return jsonify({"error": "Answers must be a list of integers."}), 400

    if q_type == 'phq9' and len(answers) == 9:
        result = score_phq9(answers)
    elif q_type == 'gad7' and len(answers) == 7:
        result = score_gad7(answers)
    else:
        return jsonify({"error": "Invalid questionnaire type or wrong number of answers."}), 400
        
    return jsonify(result)

# --- Main Execution ---
if __name__ == '__main__':
    # Run the server for Module 2 on port 5002
    app.run(debug=True, port=5002)  