# Typical Architechture of Classification Model.

import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import make_circles

# Make 1000 examples
n_samples = 1000

#create circles
x, y = make_circles(n_samples, noise=0.03, random_state=42)

# Check dimension


model = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(224, 224, 3)),
    keras.layers.Flatten(),
    keras.layers.Dense(10, activation='relu', name='dense_hidden1'),
    keras.layers.Dense(10, activation='relu', name='dense_hidden2'),
    keras.layers.Dense(3, activation='softmax')
])

# Binary classification: Sigmoid, binary_crossentropy
model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.SGD(learning_rate=1e-4), metrics=['accuracy'])