import pandas as pd
import re
import os

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
RAW_FILE = os.path.join(DATA_DIR, "raw_telegram_data.csv")
PREPROCESSED_FILE = os.path.join(DATA_DIR, "preprocessed_telegram_data.csv")

def clean_amharic_text(text):
    # Remove URLs
    text = re.sub(r"http\S+", "", str(text))
    # Remove non-Amharic chars, keep common punctuation
    text = re.sub(r"[^\u1200-\u137F\s፡።፣፤]", "", text)
    # Normalize multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize_amharic_text(text):
    # Basic whitespace tokenization (for advanced, integrate stanza, etc.)
    return text.split()

def preprocess():
    df = pd.read_csv(RAW_FILE)
    df["clean_text"] = df["text"].astype(str).apply(clean_amharic_text)
    df["tokens"] = df["clean_text"].apply(tokenize_amharic_text)
    # Reorder columns for clarity
    cols = ["msg_id", "channel", "sender", "date", "photo", "text", "clean_text", "tokens"]
    df = df[[c for c in cols if c in df.columns]]
    df.to_csv(PREPROCESSED_FILE, index=False)
    print(f"Preprocessed data saved to {PREPROCESSED_FILE}")

if __name__ == "__main__":
    preprocess()