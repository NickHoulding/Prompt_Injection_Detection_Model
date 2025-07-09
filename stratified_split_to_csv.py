import pandas as pd
import os
from sklearn.model_selection import train_test_split
from datasets import load_dataset

dataset = load_dataset("jayavibhav/prompt-injection-safety")
X = dataset["train"]["text"] + dataset["test"]["text"]
y = dataset["train"]["label"] + dataset["test"]["label"]

y = [1 if label == 2 else label for label in y]

print("Binary label distribution after conversion:")
label_counts = {i: y.count(i) for i in set(y)}
print(f"Label counts: {label_counts}")
print(f"Total samples: {len(y)}")
print()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

train_df = pd.DataFrame({"text": X_train, "label": y_train})
test_df = pd.DataFrame({"text": X_test, "label": y_test})

save_path = os.path.join(os.path.dirname(__file__), "data", "stratified_split")
os.makedirs(save_path, exist_ok=True)
train_df.to_csv(os.path.join(save_path, "train_split.csv"), index=False)
test_df.to_csv(os.path.join(save_path, "test_split.csv"), index=False)

train_dist = {i: y_train.count(i)/len(y_train) for i in set(y_train)}
test_dist = {i: y_test.count(i)/len(y_test) for i in set(y_test)}
print("Train distribution:", train_dist)
print("Test distribution:", test_dist)