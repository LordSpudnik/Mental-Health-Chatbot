import pandas as pd
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import pipeline
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='transformers')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s']", '', text)
    tokens = text.split()
    processed_tokens = [
        lemmatizer.lemmatize(word) for word in tokens if word not in stop_words
    ]
    return ' '.join(processed_tokens)

# --- Main Analysis Script ---
if __name__ == "__main__":
    processed_data_path = '../data/processed/processed_suicide_data.csv'
    output_results_path = '../data/results/analysis_results.csv'

    # Train Your Custom Risk Classifier
    print("--- Training Custom Risk Classifier ---")
    try:
        df = pd.read_csv(processed_data_path)
        df.dropna(subset=['processed_text', 'class'], inplace=True)
        
        vectorizer = TfidfVectorizer(max_features=5000)
        X_tfidf = vectorizer.fit_transform(df['processed_text'])
        y = df['class']
        
        custom_model = LogisticRegression(max_iter=1000)
        custom_model.fit(X_tfidf, y)
        print("Custom classifier trained successfully.")
        
    except FileNotFoundError:
        print(f"Error: Processed data not found at {processed_data_path}. Please run the preprocessor first.")
        exit()

    # Load the Pre-trained General Sentiment Pipeline
    print("\n--- Loading General Sentiment Model ---")
    sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
    print("General sentiment model loaded.")

    # Create the Combined Analysis Function
    def analyze_text_holistically(text_input):
        # Analyzes text using both models and returns the desired JSON format.
        sentiment_result = sentiment_pipeline(text_input)[0]
        sentiment_label = sentiment_result['label'].lower()
        confidence = round(sentiment_result['score'], 2)

        # Ensure sentiment_label is one of 'positive', 'negative', 'neutral'
        if sentiment_label not in ['positive', 'negative', 'neutral']:
            # Fallback: If model returns something unexpected, treat as neutral
            sentiment_label = 'neutral'

        processed_input = preprocess_text(text_input)
        vectorized_input = vectorizer.transform([processed_input])
        custom_prediction = custom_model.predict(vectorized_input)[0]

        if sentiment_label == 'negative':
            tone = 'high-risk' if custom_prediction == 'suicide' else 'low-risk'
        else:
            tone = 'low-risk'
            
        return {
            "text": text_input,
            "sentiment": sentiment_label,
            "tone": tone,
            "confidence_score": confidence
        }

    # Run Analysis and Collect Results
    print("\n--- Running Analysis on New Sentences ---")
    
    example_texts = [
        "I'm feeling really down and hopeless today, I don't think I can go on.",
        "I need help, everything just feels so pointless.",
        "Finally finished my project, what a relief! It was a good day.",
        "I am so tired of everything.",
        "This is the best moment of my life, I am so happy!",
        "I'm not sure if this is worth it anymore.",
        "I'm bored.",
        "I'm good.",
        "I'm quitting.",
        "I went to the store to buy some groceries."
    ]
    
    # Analyze each sentence and store the result in a list
    results_list = []
    for text in example_texts:
        result = analyze_text_holistically(text)
        results_list.append(result)
    
    results_df = pd.DataFrame(results_list)
    print(results_df)
    
    results_df.to_csv(output_results_path, index=False)
    
    print("Analysis complete. Results saved successfully!")