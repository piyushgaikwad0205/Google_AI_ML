import tensorflow as tf
import numpy as np
from tensorflow import keras

# Define the model
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

# Compile the model
model.compile(optimizer='sgd', loss='mean_squared_error')

# Define the input and output data
xs = np.array([1, 2, 3, 4, 5, 6, 7], dtype=int)
ys = np.array([.500 , 1.00], dtype=int)

# Train the model
model.fit(xs, ys, epochs=500)

# Make a prediction
print(model.predict(np.array([100])))
