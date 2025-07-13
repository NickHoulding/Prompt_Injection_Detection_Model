import numpy as np
import copy
from typing import Union

def initialize_with_zeros(dim: int) -> tuple:
    """
    Creates a vector of zeros of shape (dim, 1) for w and 
    initializes b to 0.
    
    Args:
    - dim (int): Size of the w vector we want (or number of 
        parameters in this case).

    Returns:
    - tuple (w, b):
        - w (numpy.ndarray): Initialized vector of shape (dim, 1).
        - b (float): Initialized scalar (corresponds to the bias) 
            of type float.
    """
    w = np.zeros((dim, 1))
    b = 0.0

    return w, b

def sigmoid(z: Union[float, np.ndarray]) -> np.ndarray:
    """
    Computes the sigmoid of z.

    Args:
    - z (Union[float, np.ndarray]): A scalar or numpy array of 
        any size.

    Returns:
    - s (np.ndarray): The sigmoid of z.
    """
    return 1 / (1 + np.exp(-z))

def propagate(
        w: np.ndarray, 
        b: float, 
        X: np.ndarray, 
        Y: np.ndarray
    ) -> tuple:
    """
    Implement the cost function and its gradient for propagation.

    Args:
    - w (np.ndarray): Weights, a numpy array of size (num_px * num_px * 3, 1).
    - b (float): Bias, a scalar.
    - X (np.ndarray): Data of size (num_px * num_px * 3, number of examples).
    - Y (np.ndarray): True "label" vector of size (1, number of examples).

    Returns:
    - grads (dict): Dictionary containing the gradients of the weights and bias.
        - dw (np.ndarray): Gradient of the loss with respect to w, thus 
            same shape as w.
        - db (np.ndarray): Gradient of the loss with respect to b, thus 
            same shape as b.
    - cost (float): Negative log-likelihood cost for logistic regression.
    """
    m = X.shape[1]
    
    # Forward propagation
    Z = np.dot(w.T, X) + b
    A = sigmoid(Z)
    cost = -(1/m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

    # Backward propagation
    dZ = A - Y
    dw = 1/m * np.dot(X, dZ.T)
    db = 1/m * np.sum(dZ)

    cost = np.squeeze(np.array(cost))
   
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost

def optimize(
        w: np.ndarray, 
        b: float, 
        X: np.ndarray, 
        Y: np.ndarray, 
        num_iterations=100, 
        learning_rate=0.009, 
        print_cost=False
    ) -> tuple:
    """
    Optimizes w and b by running a gradient descent algorithm.
    
    Args:
    - w (np.ndarray): Weights, a numpy array of size (num_px * num_px * 3, 1).
    - b (float): Bias, a scalar.
    - X (np.ndarray): Data of shape (num_px * num_px * 3, number of examples).
    - Y (np.ndarray): True "label" vector (containing 0 if non-cat, 1 if cat), 
        of shape (1, number of examples).
    - num_iterations (int): Number of iterations of the optimization loop.
    - learning_rate (float): Learning rate of the gradient descent update rule.
    - print_cost (bool): True to print the loss every 100 steps.

    Returns:
    - params (dict): Dictionary containing the weights w and bias b.
        - w (np.ndarray): Updated weights, a numpy array of 
            size (num_px * num_px * 3, 1).
        - b (float): Updated bias, a scalar.
    - grads (dict): Dictionary containing the gradients of the weights and 
        bias with respect to the cost function.
        - dw (np.ndarray): Gradient of the loss with respect to w, thus 
            same shape as w.
        - db (np.ndarray): Gradient of the loss with respect to b, thus 
            same shape as b.
    - costs (list): List of all the costs computed during the optimization, 
        this will be used to plot the learning curve.
    """
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)
    
    costs = []
    
    for i in range(num_iterations):
        # Cost and gradient calculation
        grads, cost = propagate(w, b, X, Y)
        
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        
        # Update rule
        w = w - (learning_rate * dw)
        b = b - (learning_rate * db)
        
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
        
            # Print the cost every 100 training iterations
            if print_cost:
                print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs

def predict(
        w: np.ndarray, 
        b: float, 
        X: np.ndarray
    ) -> np.ndarray:
    """
    Predicts whether the label is 0 or 1 using learned logistic 
    regression parameters (w, b).
    
    Args:
    - w (np.ndarray): Weights, a numpy array of size (num_px * num_px * 3, 1).
    - b (float): Bias, a scalar.
    - X (np.ndarray): Data of size (num_px * num_px * 3, number of examples).

    Returns:
    - Y_prediction (np.ndarray): A numpy array (vector) containing all 
        predictions (0/1) for the examples in X.
    """
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    
    # Compute vector A predicting the probabilities
    A = sigmoid(np.dot(w.T, X) + b)
    
    for i in range(A.shape[1]):
        # Convert probabilities A[0,i] to actual predictions p[0,i]
        if A[0, i] > 0.5:
            Y_prediction[0, i] = 1
        else:
            Y_prediction[0, i] = 0
    
    return Y_prediction

def model(
        X_train: np.ndarray, 
        Y_train: np.ndarray, 
        X_test: np.ndarray, 
        Y_test: np.ndarray, 
        num_iterations=2000, 
        learning_rate=0.5, 
        print_cost=False
    ) -> dict:
    """
    Builds the logistic regression model.

    Args:
    - X_train (np.ndarray): Training set represented by a numpy array
        of shape (num_px * num_px * 3, m_train).
    - Y_train (np.ndarray): Training labels represented by a numpy array
        (vector) of shape (1, m_train).
    - X_test (np.ndarray): Test set represented by a numpy array
        of shape (num_px * num_px * 3, m_test).
    - Y_test (np.ndarray): Test labels represented by a numpy array 
        (vector) of shape (1, m_test).
    - num_iterations (int): Hyperparameter representing the number 
        of iterations to optimize the parameters.
    - learning_rate (float): Hyperparameter representing the learning 
        rate used in the update rule of optimize().
    - print_cost (bool): Set to True to print the cost every 100 
        iterations.

    Returns:
    - d (dict): Dictionary containing information about the model.
        - costs (list): List of costs computed during the optimization.
        - Y_prediction_test (np.ndarray): Predictions on the test set.
        - Y_prediction_train (np.ndarray): Predictions on the training set.
        - w (np.ndarray): Weights learned by the model.
        - b (float): Bias learned by the model.
        - learning_rate (float): Learning rate used in the optimization.
        - num_iterations (int): Number of iterations used in the optimization.
    """
    # Initialize parameters with zeros
    # Use the "shape" function to get the first dimension of X_train
    w, b = initialize_with_zeros(dim=X_train.shape[0])
    
    # Gradient descent 
    params, grads, costs = optimize(
        w=w, 
        b=b, 
        X=X_train, 
        Y=Y_train, 
        num_iterations=num_iterations, 
        learning_rate=learning_rate, 
        print_cost=True
    )
    
    # Retrieve parameters w and b from dictionary "params"
    w, b = params["w"], params["b"]

    # Predict test/train set examples
    Y_prediction_test = predict(
        w=w, 
        b=b, 
        X=X_test
    )
    Y_prediction_train = predict(
        w=w, 
        b=b, 
        X=X_train
    )

    # Print train/test Errors
    if print_cost:
        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d