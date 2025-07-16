import numpy as np
import time
import os
from lr_model import LogisticRegressionModel

LOAD_PATH = os.path.join(os.path.dirname(__file__), 'data', 'embedded')

def main():
    X_train = np.load(os.path.join(LOAD_PATH, 'X_train.npy'))
    Y_train = np.load(os.path.join(LOAD_PATH, 'Y_train.npy'))
    X_test = np.load(os.path.join(LOAD_PATH, 'X_test.npy'))
    Y_test = np.load(os.path.join(LOAD_PATH, 'Y_test.npy'))

    model = LogisticRegressionModel(
        name="test_model",
        learning_rate=23.75,
        num_iterations=2500
    )

    tick = time.time()
    model.fit(
        X_train=X_train, 
        Y_train=Y_train, 
        print_cost=True
    )
    tock = time.time()
    seconds = tock - tick
    print(f"Training time: {seconds // 60:.2f} minutes, {seconds % 60:.2f} seconds")

    evaluation = model.evaluate(
        X_test=X_test, 
        Y_test=Y_test
    )

    print(f"Test accuracy: {evaluation['accuracy']:.2f}%")
    model.save_model(os.path.join(os.path.dirname(__file__)))

if __name__ == "__main__":
    main()