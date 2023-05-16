# Improving Computer Vision MNIST Accuracy using Convolutions

from tensorflow import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#Load data
mnist = keras.datasets.mnist
(train_images, traint_labels), (test_images, test_labels) = mnist.load_data()

# Pre Processing the data

def reshape_and_normalize(images):
    # Reshape the images to add an extra dimension
    images = np.reshape(images, (images.shape[0], images.shape[1], images.shape[2], 1))

    # Normalize pixel values
    images = images / 255.0

    return images

def build_CONV_model(callbacks):

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),


        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    # Hier you can use tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), if you did not specifie
    # the activation layer on the output layer.
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
    model.summary()
    print(f'\nCONV MODEL TRAINING')
    history = model.fit(train_images, traint_labels, epochs=10, callbacks=[callbacks])
    return model, history

#Noramalize the pixels values
train_images = train_images/255.0
test_images = test_images/255.0

if __name__ == '__main__':

    images = reshape_and_normalize(train_images)

    # Callback accuracy 99%
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if (logs.get('accuracy') >= 0.995):
                print("\n Reached 99% accuracy, so cancelling training!")
                self.model.stop_training = True

    callbacks = myCallback()

    model, history = build_CONV_model(callbacks)

    print(f'The model was trained for {len(history.epoch)}')
