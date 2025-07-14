import pickle as pkl
import numpy as np
import os
from logistic_regression_model import LogisticRegressionModel

LOAD_PATH = os.path.join(os.path.dirname(__file__), 'data', 'embedded')

def main():
    X_train = np.load(os.path.join(LOAD_PATH, 'X_train.npy'))
    Y_train = np.load(os.path.join(LOAD_PATH, 'Y_train.npy'))
    X_test = np.load(os.path.join(LOAD_PATH, 'X_test.npy'))
    Y_test = np.load(os.path.join(LOAD_PATH, 'Y_test.npy'))

    model = LogisticRegressionModel(
        learning_rate=23.75,
        num_iterations=2500
    )

    model.fit(
        X_train=X_train, 
        Y_train=Y_train, 
        print_cost=True
    )

    evaluation = model.evaluate(
        X_test=X_test, 
        Y_test=Y_test
    )

    print(f"Test accuracy: {evaluation['accuracy']:.2f}%")

    model.save_model(os.path.join(os.path.dirname(__file__), 'logistic_regression_model.pkl'))

if __name__ == "__main__":
    main()