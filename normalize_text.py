import pandas as pd
import unicodedata
import nltk
import re
import os
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

LOAD_PATH = os.path.join(os.path.dirname(__file__), "data", "stratified_split")
SAVE_PATH = os.path.join(os.path.dirname(__file__), "data", "normalized_split")
TRAIN_DF = pd.read_csv(os.path.join(LOAD_PATH, "train_split.csv"))
TEST_DF = pd.read_csv(os.path.join(LOAD_PATH, "test_split.csv"))
LEMMATIZER = WordNetLemmatizer()

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')

def normalize_text(text):
    text = text.lower()
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'[^a-z0-9\s.,;:!?-]', '', text)
    text = ' '.join(text.split())

    tokens = word_tokenize(text)
    tokens = [LEMMATIZER.lemmatize(token) if not token.isdigit() and not re.match(r'[\W\d]+', token) else token for token in tokens]
    
    return ' '.join(tokens)

def main():
    TRAIN_DF['text'] = TRAIN_DF['text'].apply(normalize_text)
    TEST_DF['text'] = TEST_DF['text'].apply(normalize_text)

    os.makedirs(SAVE_PATH, exist_ok=True)
    TRAIN_DF.to_csv(os.path.join(SAVE_PATH, "train_split_normalized.csv"), index=False)
    TEST_DF.to_csv(os.path.join(SAVE_PATH, "test_split_normalized.csv"), index=False)

if __name__ == "__main__":
    main()