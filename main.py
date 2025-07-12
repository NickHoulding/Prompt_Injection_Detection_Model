import pandas as pd
import numpy as np
import torch
import os
from sentence_transformers import SentenceTransformer
from logistic_regression import model

encoder = SentenceTransformer('all-MiniLM-L6-v2')

device = "cuda" if torch.cuda.is_available() else "cpu"
encoder.to(device)

train_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'normalized_split', 'train_split_normalized.csv'))
test_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'normalized_split', 'test_split_normalized.csv'))

# Remove rows with NaN text values
train_df = train_df.dropna(subset=['text'])
test_df = test_df.dropna(subset=['text'])
print(f"After removing NaN values - Train: {len(train_df)} samples, Test: {len(test_df)} samples")

X_train = encoder.encode(train_df['text'].tolist()).T
Y_train = np.array(train_df['label']).reshape(1, -1)
X_test = encoder.encode(test_df['text'].tolist()).T
Y_test = np.array(test_df['label']).reshape(1, -1)

print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)
print("X_test shape:", X_test.shape)
print("Y_test shape:", Y_test.shape)

d = model(
    X_train=X_train, 
    Y_train=Y_train, 
    X_test=X_test, 
    Y_test=Y_test, 
    num_iterations=2000, 
    learning_rate=0.5, 
    print_cost=True
)