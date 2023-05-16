# Steps in improving a model with TensorFlow part 1
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

x = np.array([-7., -4., -1., 3., 5., 8., 11., 14.])
y = np.array([3., 6., 9., 12., 15., 18., 21., 24.])

x = tf.constant(x)
y = tf.constant(y)

x = tf.range(-100, 100, 4)
y = x+10
plt.scatter(x, y)
plt.show()


print(x)

model = keras.models.Sequential([
    keras.layers.Dense(100, activation='relu', input_shape=(1,)),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(1)
])
model.summary()

model.compile(loss='mae', optimizer=keras.optimizers.Adam(learning_rate=0.01),metrics=['mae'])
history = model.fit(x, x, epochs=100)

y_pred = model.predict([17.0])
print(y_pred)
