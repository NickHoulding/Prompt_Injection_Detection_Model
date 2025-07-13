import pandas as pd
import os
from sklearn.model_selection import train_test_split
from datasets import load_dataset

DATASET = load_dataset("jayavibhav/prompt-injection-safety")
SAVE_PATH = os.path.join(os.path.dirname(__file__), "data", "stratified_split")

def main():
    X = DATASET["train"]["text"] + DATASET["test"]["text"]
    y = DATASET["train"]["label"] + DATASET["test"]["label"]
    y = [1 if label == 2 else label for label in y]

    print("Binary label distribution after conversion:")
    label_counts = {i: y.count(i) for i in set(y)}
    print(f"Label counts: {label_counts}")
    print(f"Total samples: {len(y)}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=0.2, 
        stratify=y, 
        random_state=42
    )

    train_df = pd.DataFrame({"text": X_train, "label": y_train})
    test_df = pd.DataFrame({"text": X_test, "label": y_test})

    os.makedirs(SAVE_PATH, exist_ok=True)
    train_df.to_csv(os.path.join(SAVE_PATH, "train_split.csv"), index=False)
    test_df.to_csv(os.path.join(SAVE_PATH, "test_split.csv"), index=False)

    train_dist = {i: y_train.count(i)/len(y_train) for i in set(y_train)}
    test_dist = {i: y_test.count(i)/len(y_test) for i in set(y_test)}
    print("Train distribution:", train_dist)
    print("Test distribution:", test_dist)

if __name__ == "__main__":
    main()