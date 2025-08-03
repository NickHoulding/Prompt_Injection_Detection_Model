import tensorflow.keras.layers as tfl
import tensorflow as tf
import numpy as np
import time
import os
from tensorflow.keras import regularizers

LOAD_PATH = os.path.join(os.path.dirname(__file__), 'embeddings')

X_train = np.load(os.path.join(LOAD_PATH, 'X_train.npy'))
Y_train = np.load(os.path.join(LOAD_PATH, 'Y_train.npy'))
X_test = np.load(os.path.join(LOAD_PATH, 'X_test.npy'))
Y_test = np.load(os.path.join(LOAD_PATH, 'Y_test.npy'))

X_train = X_train.T
X_test = X_test.T
Y_train = Y_train.flatten()
Y_test = Y_test.flatten()

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

print(X_train.shape, Y_train.shape)

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

tf.keras.models.save_model(
    model, 
    os.path.join(
        os.path.dirname(__file__), 
        'models', 
        'nn_model.keras'
    )
)
