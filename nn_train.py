import tensorflow.keras.layers as tfl
import tensorflow as tf
import numpy as np
import argparse
import time
import os
from tensorflow.keras import regularizers

# Globals
LOAD_PATH = os.path.join(os.path.dirname(__file__), 'embeddings')

def parse_args() -> argparse.Namespace:
    """
    Parses command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Train a neural network model.")
    parser.add_argument(
        '--save_model', 
        action='store_true',
        help='Whether to save the trained model.'
    )
    parser.add_argument(
        '--model_name', 
        type=str, 
        default='nn_model_' + str(time.time()), 
        help='Name of the model file to save (should be a valid filename).'
    )
    return parser.parse_args()

def train(args: argparse.Namespace) -> None:
    """
    Trains the neural network model.

    Args:
        args (argparse.Namespace): Command line arguments.
    """
    X_train = np.load(os.path.join(LOAD_PATH, 'X_train.npy'))
    Y_train = np.load(os.path.join(LOAD_PATH, 'Y_train.npy'))
    X_test = np.load(os.path.join(LOAD_PATH, 'X_test.npy'))
    Y_test = np.load(os.path.join(LOAD_PATH, 'Y_test.npy'))

    X_train = X_train.T
    X_test = X_test.T
    Y_train = Y_train.flatten()
    Y_test = Y_test.flatten()

    print(X_train.shape, Y_train.shape)

    # Model Architecture Definition
    model = tf.keras.Sequential([
        tfl.Dense(
            128, 
            activation='relu',
            kernel_regularizer=regularizers.l2(0.02),
            input_shape=(X_train.shape[1],)
        ),
        tfl.BatchNormalization(),
        tfl.Dropout(0.4),
        tfl.Dense(
            64, 
            activation='relu',
            kernel_regularizer=regularizers.l2(0.02)
        ),
        tfl.BatchNormalization(),
        tfl.Dropout(0.5),
        tfl.Dense(
            32, 
            activation='relu',
            kernel_regularizer=regularizers.l2(0.02)
        ),
        tfl.BatchNormalization(),
        tfl.Dropout(0.4),
        tfl.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    tick = time.time()
    model.fit(
        X_train, 
        Y_train, 
        epochs=100, 
        batch_size=512, 
        verbose=0
    )
    tock = time.time()

    print(f"Took: {tock - tick:.2f} seconds to train.")

    loss, accuracy = model.evaluate(
        X_train, 
        Y_train, 
        verbose=0
    )
    print(f"Train accuracy: {accuracy:.4f}")
    print(f"Loss: {loss:.4f}")

    loss, accuracy = model.evaluate(
        X_test, 
        Y_test, 
        verbose=0
    )

    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Loss: {loss:.4f}")

    if args.save_model:
        tf.keras.models.save_model(
            model, 
            os.path.join(
                os.path.dirname(__file__), 
                'models', 
                args.model_name + '.keras'
            )
        )

# Entry point
if __name__ == "__main__":
    args = parse_args()
    train(args)
