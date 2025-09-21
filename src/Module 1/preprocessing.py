# src/module1/preprocessing.py
import re
import pandas as pd
import spacy
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def clean_text(text: str) -> str:
    text = str(text)
    text = text.lower()
    text = re.sub(r'http\S+', ' ', text)                # remove urls
    text = re.sub(r'\@\w+', ' ', text)                  # remove mentions
    text = re.sub(r'[^a-z0-9\s\.\,\'\"]', ' ', text)    # keep basic punctuation
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_df(df: pd.DataFrame, text_col="text") -> pd.DataFrame:
    df = df.copy()
    df['clean_text'] = df[text_col].astype(str).apply(clean_text)
    tokens, lemmas = [], []
    for doc in nlp.pipe(df['clean_text'].tolist(), batch_size=32):
        toks = [t.text for t in doc if not t.is_stop and not t.is_punct]
        lms = [t.lemma_ for t in doc if not t.is_stop and not t.is_punct]
        tokens.append(" ".join(toks))
        lemmas.append(" ".join(lms))
    df['tokens'] = tokens
    df['lemmas'] = lemmas
    return df

if __name__ == "__main__":
    src = "data/raw/sample_texts.csv"
    dst = "data/processed/preprocessed_sample.csv"
    df = pd.read_csv(src)
    df2 = preprocess_df(df, text_col="text")
    df2.to_csv(dst, index=False)
    print(f"Saved preprocessed data -> {dst}")
