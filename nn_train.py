import tensorflow.keras.layers as tfl
import tensorflow as tf
import argparse
import time
import os
from tensorflow.keras import regularizers

from common import MODELS_PATH, f1_score, load_embeddings, report_metrics, resolve_model_path

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
    parser.add_argument(
        '--load_model',
        type=str,
        default=None,
        help='Filepath to an existing .keras model to evaluate instead of '
             'training a new one.'
    )
    return parser.parse_args()

def train(args: argparse.Namespace) -> None:
    """
    Trains the neural network model.

    Args:
        args (argparse.Namespace): Command line arguments.
    """
    X_train, Y_train, X_test, Y_test = load_embeddings()

    X_train = X_train.T
    X_test = X_test.T
    Y_train = Y_train.flatten()
    Y_test = Y_test.flatten()

    if args.load_model:
        model_path = resolve_model_path(args.load_model, expected_suffix='.keras')
        if model_path is None:
            return

        try:
            model = tf.keras.models.load_model(model_path)
        except Exception as e:
            print(f"[✗] Could not load '{model_path}' as a Keras model: {e}")
            return

        print(f"Model loaded from {model_path}")

        # Recompile with the eval metrics so evaluate() returns recall/precision
        # regardless of how the saved model was originally compiled.
        model.compile(
            loss='binary_crossentropy',
            metrics=[
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.Precision(name='precision')
            ]
        )

        _, recall, precision = model.evaluate(
            X_test,
            Y_test,
            verbose=0
        )
        report_metrics("Test", recall, f1_score(recall, precision), precision)
        return

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
        metrics=[
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.Precision(name='precision')
        ]
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

    _, recall, precision = model.evaluate(
        X_train,
        Y_train,
        verbose=0
    )
    report_metrics("Train", recall, f1_score(recall, precision), precision)

    _, recall, precision = model.evaluate(
        X_test,
        Y_test,
        verbose=0
    )
    report_metrics("Test", recall, f1_score(recall, precision), precision)

    if args.save_model:
        tf.keras.models.save_model(
            model,
            os.path.join(MODELS_PATH, args.model_name + '.keras')
        )

# Entry point
if __name__ == "__main__":
    args = parse_args()
    train(args)
