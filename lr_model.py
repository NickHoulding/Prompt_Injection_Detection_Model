import pickle as pkl
import numpy as np
import os
from typing import Union

def load_model(file_path: str) -> 'LogisticRegressionModel':
    """
    Load a model from a file.
    
    Args:
    - file_path (str): Path to the model file.
    
    Returns:
    - model (LogisticRegressionModel): Loaded model instance.
    """
    with open(file_path, 'rb') as f:
        model = pkl.load(f)
    
    print(f"Model loaded from {file_path}")
    
    return model

class LogisticRegressionModel:
    def __init__(self, learning_rate=0.01, num_iterations=2000, name="default_name"):
        self.name = name
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        self.is_trained = False
        self.costs = []
    
    def _initialize_with_zeros(self, dim: int) -> None:
        """
        Creates a vector of zeros of shape (dim, 1) for weights and 
        initializes bias to 0.
        
        Args:
        - dim (int): Size of the weight vector we want (number of features).
        """
        self.weights = np.zeros((dim, 1))
        self.bias = 0.0
    
    def _sigmoid(self, z: Union[float, np.ndarray]) -> np.ndarray:
        """
        Computes the sigmoid of z.

        Args:
        - z (Union[float, np.ndarray]): A scalar or numpy array of any size.

        Returns:
        - s (np.ndarray): The sigmoid of z.
        """
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))
    
    def _propagate(self, X: np.ndarray, Y: np.ndarray) -> tuple:
        """
        Implement the cost function and its gradient for propagation.

        Args:
        - X (np.ndarray): Data of size (num_features, number of examples).
        - Y (np.ndarray): True "label" vector of size (1, number of examples).

        Returns:
        - grads (dict): Dictionary containing the gradients of the weights and bias.
        - cost (float): Negative log-likelihood cost for logistic regression.
        """
        if self.weights is None or self.bias is None:
            raise ValueError("Weights and bias are not initialized.")
        
        m = X.shape[1]
        
        # Forward propagation
        Z = np.dot(self.weights.T, X) + self.bias
        A = self._sigmoid(Z)

        epsilon = 1e-15
        A = np.clip(A, epsilon, 1 - epsilon)

        cost = -(1/m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

        # Backward propagation
        dZ = A - Y
        dw = 1/m * np.dot(X, dZ.T)
        db = 1/m * np.sum(dZ)

        cost = np.squeeze(np.array(cost))
        grads = {"dw": dw, "db": db}
        
        return grads, cost
    
    def _optimize(
            self, 
            X: np.ndarray, 
            Y: np.ndarray, 
            print_cost=False
        ) -> None:
        """
        Optimizes weights and bias by running gradient descent algorithm.
        
        Args:
        - X (np.ndarray): Data of shape (num_features, number of examples).
        - Y (np.ndarray): True "label" vector of shape (1, number of examples).
        - print_cost (bool): True to print the loss every 100 steps.
        """
        self.costs = []
        
        for i in range(self.num_iterations):
            grads, cost = self._propagate(X, Y)
            
            dw = grads["dw"]
            db = grads["db"]
            
            self.weights = self.weights - (self.learning_rate * dw)
            self.bias = self.bias - (self.learning_rate * db)
            
            if i % 100 == 0:
                self.costs.append(cost)
            
                if print_cost:
                    print("Cost after iteration %i: %f" % (i, cost))
    
    def fit(
            self, 
            X_train: np.ndarray, 
            Y_train: np.ndarray, 
            print_cost=False
        ) -> None:
        """
        Train the logistic regression model.

        Args:
        - X_train (np.ndarray): Training set of shape (num_features, m_train).
        - Y_train (np.ndarray): Training labels of shape (1, m_train).
        - print_cost (bool): Set to True to print the cost every 100 iterations.
        """
        # Initialize parameters with zeros
        self._initialize_with_zeros(dim=X_train.shape[0])
        self._optimize(X_train, Y_train, print_cost)
        self.is_trained = True
        
        if print_cost:
            print(f"Training completed after {self.num_iterations} iterations")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts whether the label is 0 or 1 using learned logistic 
        regression parameters.
        
        Args:
        - X (np.ndarray): Data of size (num_features, number of examples).

        Returns:
        - Y_prediction (np.ndarray): A numpy array containing all 
            predictions (0/1) for the examples in X.
        """
        if not self.is_trained or self.weights is None:
            raise ValueError("Model must be trained before making predictions. Call fit() first.")
        
        weights = self.weights.reshape(X.shape[0], 1)
        A = self._sigmoid(np.dot(weights.T, X) + self.bias)
        Y_prediction = (A > 0.5).astype(int)

        return Y_prediction
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the probabilities for each class.
        
        Args:
        - X (np.ndarray): Data of size (num_features, number of examples).

        Returns:
        - probabilities (np.ndarray): Predicted probabilities for each example.
        """
        if not self.is_trained or self.weights is None:
            raise ValueError("Model must be trained before making predictions. Call fit() first.")
        
        weights = self.weights.reshape(X.shape[0], 1)

        return self._sigmoid(np.dot(weights.T, X) + self.bias)
    
    def evaluate(self, X_test: np.ndarray, Y_test: np.ndarray) -> dict:
        """
        Evaluate the model on test data.
        
        Args:
        - X_test (np.ndarray): Test set of shape (num_features, m_test).
        - Y_test (np.ndarray): Test labels of shape (1, m_test).
        
        Returns:
        - results (dict): Dictionary containing evaluation metrics.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation. Call fit() first.")
        
        Y_prediction = self.predict(X_test)
        accuracy = np.mean(Y_prediction == Y_test) * 100
        
        return {
            "model_name": self.name,
            "accuracy": accuracy,
            "predictions": Y_prediction,
            "num_correct": np.sum(Y_prediction == Y_test),
            "num_total": Y_test.shape[1]
        }
    
    def get_costs(self) -> list:
        """
        Returns the list of costs during training.
        
        Returns:
        - costs (list): List of costs computed during optimization.
        """
        return self.costs
    
    def save_model(self, file_path: str):
        """
        Save the model to a file.
        
        Args:
        - file_path (str): Path to save the model.
        """
        final_path = os.path.join(file_path, self.name + '.pkl')

        os.makedirs(file_path, exist_ok=True)
        with open(final_path, 'wb') as f:
            pkl.dump(self, f)

        print(f"Model saved to {final_path}")