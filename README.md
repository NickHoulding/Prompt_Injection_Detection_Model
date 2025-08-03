# Prompt Injection Detection Model

A machine learning project that implements multiple model architectures to detect prompt injection attacks. This project demonstrates fundamental ML concepts including data preprocessing, feature engineering with embeddings, and binary classification using both custom and deep learning approaches.

## Project Overview

This project contains a complete end-to-end implementation of a prompt injection detection system using:

- **Custom Logistic Regression**: Implemented from scratch with gradient descent optimization
- **Neural Network**: Deep learning model built with TensorFlow/Keras
- **Unified Demo Interface**: Single interactive demo supporting both models via command-line arguments
- **Streamlined Data Pipeline**: Single script handling complete data workflow from download to embeddings
- **Ollama Embeddings**: Using `nomic-embed-text` model for text embeddings
- **Stratified Data Splitting**: Ensuring balanced 80/20 train/test split
- **Text Normalization**: Comprehensive preprocessing pipeline

## Dataset

The project uses the [prompt-injection-safety](https://huggingface.co/datasets/jayavibhav/prompt-injection-safety) dataset from Hugging Face, which contains 60k labeled examples of legitimate prompts and prompt injection attempts in total.

## Model Architectures

### Logistic Regression Model
- **Algorithm**: Custom logistic regression implemented from scratch with sigmoid activation
- **Features**: 768-dimensional embeddings from Ollama's `nomic-embed-text` model
- **Optimization**: Gradient descent with learning rate 23.75 over 2500 iterations
- **Output**: Binary classification (0 = legitimate, 1 = malicious)

### Neural Network Model
- **Architecture**: Multi-layer neural network with regularization and dropout
- **Layers**: Dense layers (128, 64, 32 neurons) with batch normalization
- **Regularization**: L2 regularization and dropout for preventing overfitting
- **Framework**: TensorFlow/Keras implementation
- **Output**: Binary classification with sigmoid activation

## Project Structure

```
├── embeddings/                 # Processed embeddings ready for training
│   ├── X_train.npy             # Training feature embeddings
│   ├── X_test.npy              # Test feature embeddings  
│   ├── Y_train.npy             # Training labels
│   └── Y_test.npy              # Test labels
├── models/                     # Directory for trained model files
│   ├── lr_model.pkl            # Trained logistic regression model 
│   └── nn_model.keras          # Trained neural network model
├── data_pipeline.py            # Unified data processing pipeline
├── demo.py                     # Unified interactive demo interface
├── lr_train.py                 # Logistic regression model and training script
├── nn_train.py                 # Neural network model and training script
├── pyproject.toml              # UV project configuration
└── uv.lock                     # Dependency lock file
```

## Setup and Installation

1. **Clone the repository and navigate to the project directory:**
   ```bash
   git clone https://github.com/NickHoulding/Prompt_Injection_Detection_Model.git
   cd Prompt_Injection_Detection_Model
   ```

2. **Install uv (if not already installed):**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Install dependencies using uv:**
   ```bash
   uv sync
   ```

This project requires Python >=3.11 and uses uv for modern Python dependency management.

## Reproducing Results

The project uses a streamlined data processing pipeline for easy setup and execution:

### Step 1: Data Processing Pipeline
```bash
uv run data_pipeline.py
```
- Downloads the prompt injection dataset from Hugging Face
- Converts multi-class labels to binary (legitimate vs malicious)
- Performs stratified train/test split (80/20)
- Applies comprehensive text preprocessing (normalization, lemmatization)
- Generates embeddings using Ollama's `nomic-embed-text` model
- Saves processed embeddings as NumPy arrays to `embeddings/`
- Requires Ollama to be installed and running with the embedding model

### Step 2: Model Training

#### Logistic Regression Model
```bash
# Train with default settings and save the model
uv run lr_train.py --save_model

# Train without saving the model (for testing)
uv run lr_train.py

# Train and save with a custom model name
uv run lr_train.py --save_model --model_name my_custom_lr_model
```
- Loads processed embeddings from `embeddings/` directory
- Instantiates a new custom logistic regression model object
- Trains the model on the embedded training features
    - Hyperparameters: learning_rate=23.75, num_iterations=2500
- Evaluates model performance on the test set
- Optionally saves trained model as `models/{model_name}.pkl` when `--save_model` flag is used

#### Neural Network Model
```bash
# Train with default settings and save the model
uv run nn_train.py --save_model

# Train without saving the model (for testing)
uv run nn_train.py

# Train and save with a custom model name
uv run nn_train.py --save_model --model_name my_custom_nn_model
```
- Loads processed embeddings from `embeddings/` directory
- Creates and trains a deep neural network using TensorFlow
- Applies regularization techniques (L2, dropout, batch normalization)
- Optionally saves trained model as `models/{model_name}.keras` when `--save_model` flag is used

### Step 3: Interactive Testing
```bash
# Test with neural network model (default)
uv run demo.py

# Test with logistic regression model
uv run demo.py --model lr

# Test with neural network model (explicit)
uv run demo.py --model nn
```
- Loads the specified trained model (make sure Ollama is running and the embedding model has been pulled with `ollama pull nomic-embed-text`!)
- Provides interactive interface for testing prompts in real time
- Shows classification results with confidence scores
- Supports both model types through command-line arguments

## Command-Line Arguments

### Training Scripts

Both training scripts (`lr_train.py` and `nn_train.py`) support the following arguments:

- `--save_model`: Flag to save the trained model to disk (optional)
- `--model_name`: Custom name for the saved model file (optional, defaults to timestamped name. This *MUST* be formatted as a valid filename if specified.)

**Examples:**
```bash
# Train and save with default name
uv run lr_train.py --save_model
uv run nn_train.py --save_model

# Train and save with custom name
uv run lr_train.py --save_model --model_name custom_lr_model
uv run nn_train.py --save_model --model_name custom_nn_model

# Train without saving (useful for experimentation)
uv run lr_train.py
uv run nn_train.py
```

### Demo Script

The demo script (`demo.py`) supports:

- `--model`: Choose between 'lr' (logistic regression) or 'nn' (neural network), defaults to 'nn'

**Examples:**
```bash
# Use neural network model (default)
uv run demo.py

# Use logistic regression model
uv run demo.py --model lr

# Use neural network model (explicit)
uv run demo.py --model nn
```

## Expected Results

After training, you should see test accuracy for prompt injection detection:

- **Logistic Regression Model**: ~94% test accuracy
- **Neural Network Model**: ~95% test accuracy

Both models provide:
- Binary classification (legitimate vs malicious)
- Confidence scores (probability estimates)
- Real-time inference capabilities

## Key Features

### Multiple Model Architectures
- **Custom Logistic Regression**: Built from scratch without sklearn, includes complete model class definition and training script in a single file
- **Deep Neural Network**: TensorFlow/Keras implementation with advanced regularization
- **Unified Interface**: Single demo script supporting both models via command-line arguments
- **Flexible Training**: Command-line arguments for model saving and custom naming

### Streamlined Data Processing
- **Unified Pipeline**: Single script handles complete data workflow from download to embeddings
- **Modern Embeddings**: Ollama's `nomic-embed-text` model for high-quality representations
- **Robust Preprocessing**: Unicode normalization, lemmatization, and special character handling
- **Efficient Storage**: Direct embedding generation and storage for immediate model training

### Production-Ready Demo
- **Model Selection**: Choose between logistic regression (`--model lr`) or neural network (`--model nn`)
- **Input Validation**: Comprehensive error handling and graceful recovery
- **Interactive Interface**: Real-time prompt testing with confidence scores
- **Performance Metrics**: Detailed classification results and probability estimates

## Technical Highlights

- **Unified Data Pipeline**: Complete end-to-end processing from raw text to embeddings in a single script
- **Dual Architecture Approach**: Compare custom implementation vs. deep learning performance
- **Consolidated Model Implementation**: Logistic regression model class and training script combined for streamlined development
- **Flexible Training Options**: Command-line arguments for model saving, custom naming, and training configuration
- **Mathematical Implementation**: Custom gradient descent with vectorized operations
- **Modern Dependency Management**: UV for fast, reliable Python package management
- **Memory Efficiency**: Vectorized NumPy operations for large embedding matrices
- **Error Handling**: Comprehensive validation and graceful error recovery
- **Reproducibility**: Fixed random seeds and deterministic preprocessing

## Dependencies

Key packages in this project and how they're used:
- `ollama`: Text embedding generation
- `tensorflow`: Deep learning framework for neural network implementation
- `pandas`: Data manipulation and CSV handling
- `numpy`: Vectorized computations and array operations
- `nltk`: Natural language preprocessing (normalization)
- `datasets`: Hugging Face dataset integration
- `scikit-learn`: Dataset stratification

Project management tools:
- **UV**: Modern Python package manager (requires Python >=3.11)
- **pyproject.toml**: Modern Python project configuration

---

*This project demonstrates fundamental machine learning concepts alongside modern deep learning approaches, providing a comprehensive comparison between custom implementations and established frameworks for NLP security applications.*