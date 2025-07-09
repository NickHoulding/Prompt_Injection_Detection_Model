import pandas as pd
import unicodedata
import nltk
import re
import os
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

def normalize_text(text):
    # Lowercase
    text = text.lower()
    # Unicode normalization (NFKC)
    text = unicodedata.normalize('NFKC', text)
    # Remove irrelevant special characters (keep alphanumeric, spaces, and basic punctuation)
    text = re.sub(r'[^a-z0-9\s.,;:!?-]', '', text)
    # Trim whitespace
    text = ' '.join(text.split())
    # Tokenize and lemmatize (preserve code-like structures)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) if not token.isdigit() and not re.match(r'[\W\d]+', token) else token for token in tokens]
    
    return ' '.join(tokens)

# Load CSVs
load_path = os.path.join(os.path.dirname(__file__), "data", "stratified_split")
save_path = os.path.join(os.path.dirname(__file__), "data", "normalized_split")
train_df = pd.read_csv(os.path.join(load_path, "train_split.csv"))
test_df = pd.read_csv(os.path.join(load_path, "test_split.csv"))

# Apply normalization
train_df['text'] = train_df['text'].apply(normalize_text)
test_df['text'] = test_df['text'].apply(normalize_text)

# Save normalized CSVs
os.makedirs(save_path, exist_ok=True)
train_df.to_csv(os.path.join(save_path, "train_split_normalized.csv"), index=False)
test_df.to_csv(os.path.join(save_path, "test_split_normalized.csv"), index=False)