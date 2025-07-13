import numpy as np
import os
from logistic_regression import model

LOAD_PATH = os.path.join(os.path.dirname(__file__), 'data', 'embedded')

def main():
    X_train = np.load(os.path.join(LOAD_PATH, 'X_train.npy'))
    Y_train = np.load(os.path.join(LOAD_PATH, 'Y_train.npy'))
    X_test = np.load(os.path.join(LOAD_PATH, 'X_test.npy'))
    Y_test = np.load(os.path.join(LOAD_PATH, 'Y_test.npy'))

    d = model(
        X_train=X_train, 
        Y_train=Y_train, 
        X_test=X_test, 
        Y_test=Y_test, 
        num_iterations=2500, 
        learning_rate=23.75, 
        print_cost=True
    )

if __name__ == "__main__":
    main()