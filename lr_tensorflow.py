import tensorflow as tf
import numpy as np
import time
import os

LOAD_PATH = os.path.join(os.path.dirname(__file__), 'data', 'embedded')

X_train = np.load(os.path.join(LOAD_PATH, 'X_train.npy'))
Y_train = np.load(os.path.join(LOAD_PATH, 'Y_train.npy'))
X_test = np.load(os.path.join(LOAD_PATH, 'X_test.npy'))
Y_test = np.load(os.path.join(LOAD_PATH, 'Y_test.npy'))

X_train = X_train.T  # Transpose from (384, 47962) to (47962, 384)
X_test = X_test.T    # Transpose from (384, 11994) to (11994, 384)
Y_train = Y_train.flatten()
Y_test = Y_test.flatten()

model = tf.keras.Sequential([
    tf.keras.layers.Dense(
        1, 
        activation='sigmoid', 
        input_shape=(X_train.shape[1],)
    )
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
    epochs=50, 
    batch_size=32, 
    verbose=0
)
tock = time.time()

print(f"Took: {tock - tick:.2f} seconds to train.")

loss, accuracy = model.evaluate(
    X_test, 
    Y_test, 
    verbose=0
)
print(f"Test accuracy: {accuracy:.4f}")
print(f"Loss: {loss:.4f}")

predictions = model.predict(X_test)
predictions = (predictions > 0.5).astype(int).flatten()