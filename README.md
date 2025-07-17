# Prompt Injection Detection Model - Logistic Regression Implementation

A machine learning project that implements a custom logistic regression model from scratch to detect prompt injection attacks. This project demonstrates fundamental ML concepts including data preprocessing, feature engineering with embeddings, and binary classification.

## Project Overview

This branch (`logistic_regression_model`) contains a complete end-to-end implementation of a prompt injection detection system using:

- **Custom Logistic Regression**: Implemented from scratch with gradient descent optimization
- **Sentence Transformers**: Using `all-MiniLM-L6-v2` for text embeddings from `sentence_transformers`
- **Stratified Data Splitting**: Ensuring balanced 80/20 train/test split
- **Text Normalization**: Comprehensive preprocessing pipeline
- **Interactive Demo**: Real-time testing interface

## Dataset

The project uses the [prompt-injection-safety](https://huggingface.co/datasets/jayavibhav/prompt-injection-safety) dataset from Hugging Face, which contains 60k labeled examples of legitimate prompts and prompt injection attempts in total.

## Model Architecture

- **Algorithm**: Logistic Regression with sigmoid activation
- **Features**: 384-dimensional sentence embeddings from SentenceTransformers
- **Optimization**: Gradient descent with learning rate 23.75 over 2500 iterations
- **Output**: Binary classification (0 = legitimate, 1 = malicious)

## Project Structure

```
├── data/
│   ├── stratified_split/        # Train/test CSV files
│   ├── normalized_split/        # Preprocessed text data
│   └── embedded/                # NumPy arrays with embeddings
├── models/                      # Directory for model files
├── stratified_split_to_csv.py   # Step 1: Data loading and splitting
├── normalize_text.py            # Step 2: Text preprocessing
├── embed_text.py                # Step 3: Feature extraction
├── train.py                     # Step 4: Model training
├── demo.py                      # Step 5: Interactive testing interface
├── lr_model.py                  # Custom LR model implementation
└── requirements.txt             # Dependencies
```

## Setup and Installation

1. **Clone the repository and navigate to the project directory:**
   ```bash
   git clone https://github.com/NickHoulding/lr_prompt_injection_detection_model.git
   cd lr_prompt_injection_detection_model
   git checkout logistic_regression_model
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Reproducing Results

The project consists of 5 sequential scripts that must be run in order for proper setup:

### Step 1: Data Preparation
```bash
python stratified_split_to_csv.py
```
- Downloads the prompt injection dataset from Hugging Face
- Converts multi-class labels to binary (legitimate vs malicious)
- Performs stratified train/test split (80/20)
- Saves CSV files to `data/stratified_split/`

### Step 2: Text Normalization
```bash
python normalize_text.py
```
- Applies comprehensive text preprocessing:
  - Lowercasing and Unicode normalization
  - Special character removal
  - Tokenization and lemmatization
- Saves normalized data to `data/normalized_split/`

### Step 3: Feature Extraction
```bash
python embed_text.py
```
- Generates 384-dimensional sentence embeddings using `all-MiniLM-L6-v2`
- Converts text to numerical embedding features
- Saves NumPy arrays to `data/embedded/`
- Uses GPU acceleration if available

### Step 4: Model Training
```bash
python train.py
```
- Instantiates a new custom logistic regression model object
- Trains the model on the embedded training features
    - Hyperparameters: learning_rate=23.75, num_iterations=2500
- Evaluates model performance on the test set
- Saves trained model as `lr_model.pkl`

### Step 5: Interactive Testing
```bash
python demo.py
```
- Loads the trained model and embedding pipeline
- Provides interactive interface for testing prompts in real time
- Shows classification results with confidence scores

## Expected Results

After training (with the specified hyperparameters), you should see test accuracy around **90.18%** for prompt injection detection. The model provides:

- Binary classification (legitimate vs malicious)
- Confidence scores (probability estimates)
- Real-time inference capabilities

## Key Features

### Custom Logistic Regression Implementation
- Built from scratch without sklearn
- Includes sigmoid activation, cost function, and gradient descent
- Supports model serialization and loading
- Provides probability estimates

### Robust Text Processing
- Unicode normalization for consistent encoding
- Lemmatization for morphological analysis
- Special character handling
- Whitespace normalization

### Production-Ready Demo
- Input validation and error handling
- GPU acceleration support
- Interactive command-line interface
- Model performance metrics display

## Technical Highlights

- **Mathematical Implementation**: Custom gradient descent with vectorized operations
- **Memory Efficiency**: Optimized NumPy operations for large embedding matrices
- **Error Handling**: Comprehensive validation and graceful error recovery
- **Reproducibility**: Fixed random seeds and deterministic preprocessing

## Dependencies

Key packages used in this project:
- `sentence-transformers`: Text embedding generation
- `torch`: GPU acceleration and tensor operations
- `pandas`: Data manipulation and CSV handling
- `numpy`: Numerical computations
- `nltk`: Natural language preprocessing
- `datasets`: Hugging Face dataset integration

---

*This project demonstrates fundamental machine learning concepts and can serve as a foundation for more advanced NLP security applications.*