"""
Unified Data Processing Pipeline for the Prompt Injection Detection Model

This script combines the entire data processing workflow:
1. Get the raw dataset from HuggingFace (downloads and caches to data/ on
   first run, loads the cached copy on later runs)
2. Text normalization (cached to data/ after running)
3. Stratified train/test split
4. Text embedding using Ollama
5. Save embeddings as numpy arrays to data/embeddings/

Each expensive stage is skipped when a valid cache already exists for it, so
re-running the script after the first successful run is fast: a run with
cached embeddings does no downloading, preprocessing, or embedding at all.
"""

import pandas as pd
import numpy as np
import unicodedata
import ollama
import nltk
import re
import os
import sys
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from datasets import load_dataset

from common import EMBEDDINGS_PATH

DATA_PATH = os.path.join(os.path.dirname(__file__), "data")
RAW_DATASET_PATH = os.path.join(DATA_PATH, "raw_dataset.parquet")
PROCESSED_DATASET_PATH = os.path.join(DATA_PATH, "processed_dataset.parquet")


class DataPipeline:
    """
    Unified data processing pipeline for prompt injection detection.

    Attributes:
        embeddings_path (str): Directory to load/save the final embedding arrays.
        raw_dataset_path (str): File path for the cached raw HuggingFace dataset.
        processed_dataset_path (str): File path for the cached preprocessed dataset.
        encoder (str): Ollama model to use for text embedding.
        test_size (float): Proportion of dataset for testing.
        random_state (int): Random seed for reproducibility.
        lemmatizer (WordNetLemmatizer): NLTK lemmatizer used during text normalization.

    Methods:
        _download_nltk_data(): Downloads required NLTK datasets.
        _save_dataset(X, y, path): Saves a (text, label) dataset to disk as Parquet.
        _load_cached_embeddings(): Loads cached embeddings if they exist and load cleanly.
        get_raw_dataset(dataset_name): Loads the cached raw dataset, or downloads and caches it.
        preprocess_dataset(X, y): Normalizes text, drops empty rows, and caches the result.
        stratified_split(X, y): Performs stratified train/test split.
        normalize_text(text): Normalizes a single text sample.
        embed_texts(X_train, X_test, y_train, y_test): Generates embeddings for text data using Ollama.
        save_arrays(X_train, X_test, Y_train, Y_test): Saves the final embeddings to disk.
        run_pipeline(dataset_name): Executes the complete data processing pipeline.

    Raises:
        Exception: If any step in the pipeline fails.
    """

    def __init__(
        self,
        embeddings_path=None,
        raw_dataset_path=None,
        processed_dataset_path=None,
        encoder="nomic-embed-text",
        test_size=0.2,
        random_state=42,
    ) -> None:
        """
        Initialize the data pipeline.

        Args:
            embeddings_path (str): Directory to load/save the final embedding arrays
            raw_dataset_path (str): File path for the cached raw dataset
            processed_dataset_path (str): File path for the cached preprocessed dataset
            encoder (str): Ollama model to use for text embedding
            test_size (float): Proportion of dataset for testing
            random_state (int): Random seed for reproducibility
        """
        self.embeddings_path = embeddings_path or EMBEDDINGS_PATH
        self.raw_dataset_path = raw_dataset_path or RAW_DATASET_PATH
        self.processed_dataset_path = processed_dataset_path or PROCESSED_DATASET_PATH
        self.encoder = encoder
        self.test_size = test_size
        self.random_state = random_state
        self.lemmatizer = WordNetLemmatizer()
        self._download_nltk_data()

    def _download_nltk_data(self) -> None:
        """Download required NLTK datasets."""
        try:
            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)
            nltk.download("wordnet", quiet=True)
            print("[✓] NLTK data downloaded successfully")

        except Exception as e:
            print(f"[!] Warning: Could not download NLTK data: {e}")

    def _save_dataset(self, X: list, y: list, path: str) -> None:
        """
        Save a (text, label) dataset to disk as Parquet.

        Args:
            X (list): Text samples.
            y (list): Labels corresponding to X.
            path (str): Destination file path.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pd.DataFrame({"text": X, "label": y}).to_parquet(path)
        print(f"[✓] Cached dataset to {path}")

    def _load_cached_embeddings(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        """
        Load cached embeddings from ``embeddings_path`` if they exist and load cleanly.

        Returns:
            tuple | None: ``(X_train, Y_train, X_test, Y_test)`` if a valid cache
                was found, otherwise ``None``.
        """
        filenames = ["X_train.npy", "Y_train.npy", "X_test.npy", "Y_test.npy"]
        paths = [os.path.join(self.embeddings_path, name) for name in filenames]

        if not all(os.path.exists(p) for p in paths):
            return None

        try:
            X_train, Y_train, X_test, Y_test = (np.load(path) for path in paths)
            return X_train, Y_train, X_test, Y_test
        except Exception as e:
            print(
                f"[!] Warning: Cached embeddings at {self.embeddings_path} failed "
                f"to load ({e}); regenerating"
            )
            return None

    def get_raw_dataset(
        self, dataset_name="jayavibhav/prompt-injection-safety"
    ) -> tuple[list[str], list[int]]:
        """
        Load the cached raw dataset from disk, or download and cache it if missing.

        Args:
            dataset_name (str): Name of the dataset on HuggingFace

        Returns:
            tuple: Combined texts and labels (as lists)

        Raises:
            Exception: If dataset download or caching fails
        """
        print("[1/5] Getting raw dataset")

        if os.path.exists(self.raw_dataset_path):
            try:
                df = pd.read_parquet(self.raw_dataset_path)
                print(f"[✓] Found cached raw dataset at {self.raw_dataset_path}")
                return df["text"].tolist(), df["label"].tolist()
            except Exception as e:
                print(
                    f"[!] Warning: Cached raw dataset at {self.raw_dataset_path} "
                    f"failed to load ({e}); re-downloading"
                )

        print(f"    Downloading dataset: {dataset_name}")

        try:
            dataset = load_dataset(dataset_name)

            X = list(dataset["train"]["text"]) + list(dataset["test"]["text"])
            y = list(dataset["train"]["label"]) + list(dataset["test"]["label"])
            y = [1 if label == 2 else label for label in y]

            print(f"[✓] Dataset downloaded successfully")
            print(f"    Total samples: {len(y)}")

            label_counts = {i: y.count(i) for i in set(y)}
            print(f"    Label distribution: {label_counts}")

            self._save_dataset(X, y, self.raw_dataset_path)

            return X, y

        except Exception as e:
            print(f"[✗] Error downloading dataset: {e}")
            raise

    def normalize_text(self, text: str) -> str:
        """
        Normalize a single text sample.

        Args:
            text (str): Input text to normalize

        Returns:
            str: Normalized text

        Raises:
            Exception: If text normalization fails
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""

        text = text.lower()
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"[^a-z0-9\s.,;:!?-]", "", text)
        text = " ".join(text.split())

        try:
            tokens = word_tokenize(text)
            tokens = [
                (
                    self.lemmatizer.lemmatize(token)
                    if not token.isdigit() and not re.match(r"[\W\d]+", token)
                    else token
                )
                for token in tokens
            ]

            return " ".join(tokens)

        except Exception as e:
            print(f"[✗] Error normalizing text: {e}")
            return text

    def preprocess_dataset(self, X: list, y: list) -> tuple[list[str], list[int]]:
        """
        Normalize all text samples, drop any that become empty, and cache the result.

        Args:
            X (list): Raw text samples
            y (list): Labels corresponding to X

        Returns:
            tuple: Normalized text samples and their labels, with empty rows removed

        Raises:
            Exception: If text normalization fails
        """
        print("[2/5] Preprocessing text data")

        try:
            X_normalized = [self.normalize_text(text) for text in X]
            valid_indices = [i for i, text in enumerate(X_normalized) if text.strip()]

            X_clean = [X_normalized[i] for i in valid_indices]
            y_clean = [y[i] for i in valid_indices]

            print(f"[✓] Text preprocessing completed")
            print(f"    Samples after cleaning: {len(X_clean)}")

            self._save_dataset(X_clean, y_clean, self.processed_dataset_path)

            return X_clean, y_clean

        except Exception as e:
            print(f"[✗] Error in text preprocessing: {e}")
            raise

    def stratified_split(
        self, X: list, y: list
    ) -> tuple[list[str], list[str], list[int], list[int]]:
        """
        Perform stratified train/test split to maintain label distribution.

        Args:
            X (list): List of text samples
            y (list): List of labels

        Returns:
            tuple: X_train, X_test, y_train, y_test (as lists)

        Raises:
            Exception: If stratified split fails
        """
        print("[3/5] Performing stratified train/test split")

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self.test_size,
                stratify=y,
                random_state=self.random_state,
            )

            print(f"[✓] Stratified split completed")
            print(f"    Train samples: {len(X_train)}")
            print(f"    Test samples: {len(X_test)}")

            train_dist = {i: y_train.count(i) / len(y_train) for i in set(y_train)}
            test_dist = {i: y_test.count(i) / len(y_test) for i in set(y_test)}
            print(f"    Train distribution: {train_dist}")
            print(f"    Test distribution: {test_dist}")

            return X_train, X_test, y_train, y_test

        except Exception as e:
            print(f"[✗] Error in stratified split: {e}")
            raise

    def _embed_batched(self, texts: list, batch_size: int = 64) -> np.ndarray:
        """
        Embed a list of texts with Ollama in fixed-size batches.

        Sending the full dataset in a single request causes the Ollama model
        runner to crash (EOF on /tokenize), so requests are chunked.

        Args:
            texts (list): Text samples to embed.
            batch_size (int): Number of texts per Ollama request.

        Returns:
            np.ndarray: Embeddings of shape (num_texts, embedding_dim).
        """
        embeddings = []
        total = len(texts)

        for start in range(0, total, batch_size):
            batch = texts[start : start + batch_size]
            response = ollama.embed(model=self.encoder, input=batch)
            embeddings.extend(response["embeddings"])
            print(f"      {min(start + batch_size, total)}/{total} embedded", end="\r")

        print()
        return np.array(embeddings)

    def embed_texts(
        self, X_train: list, X_test: list, y_train: list, y_test: list
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate embeddings for text data using Ollama.

        Args:
            X_train (list): Training text samples
            X_test (list): Test text samples
            y_train (list): Training labels
            y_test (list): Test labels

        Returns:
            tuple: X_train_embedded, X_test_embedded, Y_train, Y_test (as numpy arrays)

        Raises:
            Exception: If embedding generation fails
        """
        print(f"[4/5] Generating embeddings using {self.encoder}")

        try:
            print("    Embedding training data...")
            X_train_embedded = self._embed_batched(X_train).T

            print("    Embedding test data...")
            X_test_embedded = self._embed_batched(X_test).T

            Y_train = np.array(y_train).reshape(1, -1)
            Y_test = np.array(y_test).reshape(1, -1)

            print(f"[✓] Embedding completed")
            print(f"    Training embeddings shape: {X_train_embedded.shape}")
            print(f"    Test embeddings shape: {X_test_embedded.shape}")
            print(f"    Training labels shape: {Y_train.shape}")
            print(f"    Test labels shape: {Y_test.shape}")

            return X_train_embedded, X_test_embedded, Y_train, Y_test

        except Exception as e:
            print(f"[✗] Error in embedding generation: {e}")
            raise

    def save_arrays(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        Y_train: np.ndarray,
        Y_test: np.ndarray,
    ) -> None:
        """
        Save the processed arrays to disk.

        Args:
            X_train (np.ndarray): Training feature embeddings
            X_test (np.ndarray): Test feature embeddings
            Y_train (np.ndarray): Training labels
            Y_test (np.ndarray): Test labels

        Raises:
            Exception: If saving arrays fails
        """
        print("[5/5] Saving numpy arrays to disk")

        try:
            os.makedirs(self.embeddings_path, exist_ok=True)

            np.save(os.path.join(self.embeddings_path, "X_train.npy"), X_train)
            np.save(os.path.join(self.embeddings_path, "Y_train.npy"), Y_train)
            np.save(os.path.join(self.embeddings_path, "X_test.npy"), X_test)
            np.save(os.path.join(self.embeddings_path, "Y_test.npy"), Y_test)

            print(f"[✓] Arrays saved successfully to: {self.embeddings_path}")
            print("    Files created:")
            print(f"      - X_train.npy: {X_train.shape}")
            print(f"      - Y_train.npy: {Y_train.shape}")
            print(f"      - X_test.npy: {X_test.shape}")
            print(f"      - Y_test.npy: {Y_test.shape}")

        except Exception as e:
            print(f"[✗] Error saving arrays: {e}")
            raise

    def run_pipeline(self, dataset_name="jayavibhav/prompt-injection-safety") -> None:
        """
        Execute the complete data processing pipeline.

        Skips straight to done if valid embeddings are already cached at
        ``embeddings_path``. Otherwise gets the raw dataset (cached or
        downloaded), preprocesses and caches it, splits it, and generates and
        saves embeddings.

        Args:
            dataset_name (str): Name of the dataset on HuggingFace

        Raises:
            Exception: If any step in the pipeline fails
        """
        print("=" * 60)
        print("PROMPT INJECTION DETECTION - DATA PROCESSING PIPELINE")
        print("=" * 60)

        try:
            if self._load_cached_embeddings() is not None:
                print(
                    f"[✓] Embeddings already exist at {self.embeddings_path}, "
                    "skipping pipeline"
                )
                print("=" * 60)
                return

            X, y = self.get_raw_dataset(dataset_name)
            X_clean, y_clean = self.preprocess_dataset(X, y)
            X_train, X_test, y_train, y_test = self.stratified_split(X_clean, y_clean)

            X_train_emb, X_test_emb, Y_train_final, Y_test_final = self.embed_texts(
                X_train, X_test, y_train, y_test
            )

            self.save_arrays(X_train_emb, X_test_emb, Y_train_final, Y_test_final)

            print("=" * 60)
            print("[✓] DATA PROCESSING PIPELINE COMPLETED SUCCESSFULLY!")
            print("=" * 60)

        except Exception as e:
            print("=" * 60)
            print(f"[✗] PIPELINE FAILED: {e}")
            print("=" * 60)
            raise


def main() -> None:
    """
    Main function to run the data processing pipeline.

    Exits with status 1 (without a Python traceback) if the pipeline fails;
    ``run_pipeline`` has already printed a diagnostic message by that point.
    """
    pipeline = DataPipeline(encoder="nomic-embed-text", test_size=0.2, random_state=42)

    try:
        pipeline.run_pipeline()
    except Exception:
        sys.exit(1)


# Entry point
if __name__ == "__main__":
    main()
