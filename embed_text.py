import pandas as pd
import numpy as np
import ollama
import os

SAVE_PATH = os.path.join(os.path.dirname(__file__), 'data', 'embedded')
LOAD_PATH = os.path.join(os.path.dirname(__file__), 'data', 'normalized_split')
ENCODER = 'nomic-embed-text'

def main():
    train_df = pd.read_csv(os.path.join(LOAD_PATH, 'train_split_normalized.csv'))
    test_df = pd.read_csv(os.path.join(LOAD_PATH, 'test_split_normalized.csv'))

    train_df = train_df.dropna(subset=['text'])
    test_df = test_df.dropna(subset=['text'])
    print(f"After removing NaN values - Train: {len(train_df)} samples, Test: {len(test_df)} samples")

    X_train = np.array(ollama.embed(
        model=ENCODER, 
        input=train_df['text'].tolist()
    )['embeddings']).T
    Y_train = np.array(train_df['label']).reshape(1, -1)
    
    X_test = np.array(ollama.embed(
        model=ENCODER, 
        input=test_df['text'].tolist()
    )['embeddings']).T
    Y_test = np.array(test_df['label']).reshape(1, -1)

    os.makedirs(SAVE_PATH, exist_ok=True)
    np.save(os.path.join(SAVE_PATH, 'X_train.npy'), X_train)
    np.save(os.path.join(SAVE_PATH, 'Y_train.npy'), Y_train)
    np.save(os.path.join(SAVE_PATH, 'X_test.npy'), X_test)
    np.save(os.path.join(SAVE_PATH, 'Y_test.npy'), Y_test)

if __name__ == "__main__":
    main()