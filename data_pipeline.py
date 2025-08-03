"""
Unified Data Processing Pipeline for the Prompt Injection Detection Model

This script combines the entire data processing workflow:
1. Download dataset from HuggingFace
2. Stratified train/test split
3. Text normalization
4. Text embedding using Ollama
5. Save as numpy arrays for model training
"""

import pandas as pd
import numpy as np
import unicodedata
import ollama
import nltk
import re
import os
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from datasets import load_dataset
from typing import List, Tuple

class DataPipeline:
    """
    Unified data processing pipeline for prompt injection detection.

    Attributes:
        save_path (str): Path to save the final numpy arrays.
        encoder (str): Ollama model to use for text embedding.
        test_size (float): Proportion of dataset for testing.
        random_state (int): Random seed for reproducibility.

    Methods:
        _download_nltk_data(): Downloads required NLTK datasets.
        download_dataset(dataset_name): Downloads dataset from HuggingFace.
        stratified_split(X, y): Performs stratified train/test split.
        normalize_text(text): Normalizes a single text sample.
        normalize_dataset(X_train, X_test): Normalizes text data for both train and test sets.
        embed_texts(X_train, X_test, y_train, y_test): Generates embeddings for text data using Ollama.
        save_arrays(X_train, X_test, Y_train, Y_test): Saves the processed arrays to disk.
        run_pipeline(dataset_name): Executes the complete data processing pipeline.

    Raises:
        Exception: If any step in the pipeline fails.
    """
    def __init__(
            self, 
            save_path=None, 
            encoder='nomic-embed-text', 
            test_size=0.2, 
            random_state=42
        ) -> None:
        """
        Initialize the data pipeline.
        
        Args:
            save_path (str): Path to save the final numpy arrays
            encoder (str): Ollama model to use for text embedding
            test_size (float): Proportion of dataset for testing
            random_state (int): Random seed for reproducibility
        """
        self.save_path = save_path or os.path.join(os.path.dirname(__file__), 'embeddings')
        self.encoder = encoder
        self.test_size = test_size
        self.random_state = random_state
        self.lemmatizer = WordNetLemmatizer()
        self._download_nltk_data()
    
    def _download_nltk_data(self) -> None:
        """Download required NLTK datasets."""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            nltk.download('wordnet', quiet=True)
            print("[✓] NLTK data downloaded successfully")
        
        except Exception as e:
            print(f"[!] Warning: Could not download NLTK data: {e}")
    
    def download_dataset(
            self, 
            dataset_name="jayavibhav/prompt-injection-safety"
        ) -> Tuple[List[str], List[int]]:
        """
        Download the dataset from HuggingFace.
        
        Args:
            dataset_name (str): Name of the dataset on HuggingFace
            
        Returns:
            tuple: Combined texts and labels (as lists)

        Raises:
            Exception: If dataset download fails
        """
        print(f"[1/5] Downloading dataset: {dataset_name}")
        
        try:
            dataset = load_dataset(dataset_name)
            
            X = list(dataset["train"]["text"]) + list(dataset["test"]["text"])
            y = list(dataset["train"]["label"]) + list(dataset["test"]["label"])
            y = [1 if label == 2 else label for label in y]
            
            print(f"[✓] Dataset downloaded successfully")
            print(f"    Total samples: {len(y)}")
            
            label_counts = {i: y.count(i) for i in set(y)}
            print(f"    Label distribution: {label_counts}")
            
            return X, y
            
        except Exception as e:
            print(f"[✗] Error downloading dataset: {e}")
            raise
    
    def stratified_split(
            self, 
            X: list, 
            y: list
        ) -> Tuple[List[str], List[str], List[int], List[int]]:
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
        print("[2/5] Performing stratified train/test split")
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, 
                y, 
                test_size=self.test_size, 
                stratify=y, 
                random_state=self.random_state
            )
            
            print(f"[✓] Stratified split completed")
            print(f"    Train samples: {len(X_train)}")
            print(f"    Test samples: {len(X_test)}")
            
            train_dist = {i: y_train.count(i)/len(y_train) for i in set(y_train)}
            test_dist = {i: y_test.count(i)/len(y_test) for i in set(y_test)}
            print(f"    Train distribution: {train_dist}")
            print(f"    Test distribution: {test_dist}")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            print(f"[✗] Error in stratified split: {e}")
            raise
    
    def normalize_text(
            self, 
            text: str
        ) -> str:
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
        text = unicodedata.normalize('NFKC', text)
        text = re.sub(r'[^a-z0-9\s.,;:!?-]', '', text)
        text = ' '.join(text.split())

        try:
            tokens = word_tokenize(text)
            tokens = [
                self.lemmatizer.lemmatize(token) 
                if not token.isdigit() and not re.match(r'[\W\d]+', token) 
                else token 
                for token in tokens
            ]
            
            return ' '.join(tokens)
        
        except Exception as e:
            print(f"[✗] Error normalizing text: {e}")
            return text
    
    def normalize_dataset(
            self, 
            X_train: list, 
            X_test: list
        ) -> Tuple[List[str], List[str], List[int], List[int]]:
        """
        Normalize text data for both train and test sets.
        
        Args:
            X_train (list): Training text samples
            X_test (list): Test text samples
        
        Returns:
            tuple: Normalized X_train, X_test, train_valid_indices, test_valid_indices (as lists)
        
        Raises:
            Exception: If text normalization fails
        """
        print("[3/5] Normalizing text data")
        
        try:
            X_train_normalized = [self.normalize_text(text) for text in X_train]
            X_test_normalized = [self.normalize_text(text) for text in X_test]
            
            train_valid_indices = [i for i, text in enumerate(X_train_normalized) if text.strip()]
            test_valid_indices = [i for i, text in enumerate(X_test_normalized) if text.strip()]
            
            X_train_clean = [X_train_normalized[i] for i in train_valid_indices]
            X_test_clean = [X_test_normalized[i] for i in test_valid_indices]
            
            print(f"[✓] Text normalization completed")
            print(f"    Train samples after cleaning: {len(X_train_clean)}")
            print(f"    Test samples after cleaning: {len(X_test_clean)}")
            
            return X_train_clean, X_test_clean, train_valid_indices, test_valid_indices
            
        except Exception as e:
            print(f"[✗] Error in text normalization: {e}")
            raise
    
    def embed_texts(
            self, 
            X_train: list, 
            X_test: list, 
            y_train: list, 
            y_test: list
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
            train_response = ollama.embed(
                model=self.encoder, 
                input=X_train
            )
            X_train_embedded = np.array(train_response['embeddings']).T
            
            print("    Embedding test data...")
            test_response = ollama.embed(
                model=self.encoder, 
                input=X_test
            )
            X_test_embedded = np.array(test_response['embeddings']).T
            
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
            Y_test: np.ndarray
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
            os.makedirs(self.save_path, exist_ok=True)
            
            np.save(os.path.join(self.save_path, 'X_train.npy'), X_train)
            np.save(os.path.join(self.save_path, 'Y_train.npy'), Y_train)
            np.save(os.path.join(self.save_path, 'X_test.npy'), X_test)
            np.save(os.path.join(self.save_path, 'Y_test.npy'), Y_test)
            
            print(f"[✓] Arrays saved successfully to: {self.save_path}")
            print("    Files created:")
            print(f"      - X_train.npy: {X_train.shape}")
            print(f"      - Y_train.npy: {Y_train.shape}")
            print(f"      - X_test.npy: {X_test.shape}")
            print(f"      - Y_test.npy: {Y_test.shape}")
            
        except Exception as e:
            print(f"[✗] Error saving arrays: {e}")
            raise
    
    def run_pipeline(
            self, 
            dataset_name="jayavibhav/prompt-injection-safety"
        ) -> None:
        """
        Execute the complete data processing pipeline.
        
        Args:
            dataset_name (str): Name of the dataset on HuggingFace
        
        Raises:
            Exception: If any step in the pipeline fails
        """
        print("=" * 60)
        print("PROMPT INJECTION DETECTION - DATA PROCESSING PIPELINE")
        print("=" * 60)
        
        try:
            X, y = self.download_dataset(dataset_name)
            X_train, X_test, y_train, y_test = self.stratified_split(X, y)
            X_train_norm, X_test_norm, train_indices, test_indices = self.normalize_dataset(X_train, X_test)
            
            y_train_clean = [y_train[i] for i in train_indices]
            y_test_clean = [y_test[i] for i in test_indices]
            
            X_train_emb, X_test_emb, Y_train_final, Y_test_final = self.embed_texts(
                X_train_norm, X_test_norm, y_train_clean, y_test_clean
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
    """
    pipeline = DataPipeline(
        encoder='nomic-embed-text',
        test_size=0.2,
        random_state=42
    )
    pipeline.run_pipeline()

# Entry point
if __name__ == "__main__":
    main()
