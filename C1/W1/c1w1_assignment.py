# In This exercice we will try to build a neural network that predict the price of a house

import tensorflow as tf
from tensorflow import keras
import numpy as np

def house_model():

    #Define input and output tensors with the values for houses with 1 up to 6 bedrooms
    xs = np.array([0.0, 1.0, 2.0, 3.0, 2.0, 3.0, 4.0], dtype=float)
    ys = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5], dtype=float)


    # Define model
    model = keras.Sequential([
        keras.layers.Dense(1, input_shape=[1])
    ])
    # compile
    model.compile(loss='mse', optimizer='sgd')
    model.summary()
    history = model.fit(xs, ys, epochs=1000)

    return model

if __name__ == '__main__':

    model = house_model()

    #prediction
    new_y = 7.0
    prediction = model.predict([new_y])
    print(prediction)
    print(prediction[0])

    new_y = 8.0
    prediction = model.predict([new_y])
    print(prediction)
    print(prediction[0])



