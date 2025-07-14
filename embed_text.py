import pandas as pd
import numpy as np
import torch
import os
from sentence_transformers import SentenceTransformer

SAVE_PATH = os.path.join(os.path.dirname(__file__), 'data', 'embedded')
LOAD_PATH = os.path.join(os.path.dirname(__file__), 'data', 'normalized_split')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ENCODER = SentenceTransformer('all-MiniLM-L6-v2')
ENCODER.to(DEVICE)

def main():
    train_df = pd.read_csv(os.path.join(LOAD_PATH, 'train_split_normalized.csv'))
    test_df = pd.read_csv(os.path.join(LOAD_PATH, 'test_split_normalized.csv'))

    train_df = train_df.dropna(subset=['text'])
    test_df = test_df.dropna(subset=['text'])
    print(f"After removing NaN values - Train: {len(train_df)} samples, Test: {len(test_df)} samples")

    X_train = ENCODER.encode(train_df['text'].tolist()).T
    Y_train = np.array(train_df['label']).reshape(1, -1)
    X_test = ENCODER.encode(test_df['text'].tolist()).T
    Y_test = np.array(test_df['label']).reshape(1, -1)

    os.makedirs(SAVE_PATH, exist_ok=True)
    np.save(os.path.join(SAVE_PATH, 'X_train.npy'), X_train)
    np.save(os.path.join(SAVE_PATH, 'Y_train.npy'), Y_train)
    np.save(os.path.join(SAVE_PATH, 'X_test.npy'), X_test)
    np.save(os.path.join(SAVE_PATH, 'Y_test.npy'), Y_test)

if __name__ == "__main__":
    main()