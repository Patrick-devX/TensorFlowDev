# Improving Computer Vision Accuracy using Convolutions

from tensorflow import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load the Fashion MNIST dataset
fmnist = keras.datasets.fashion_mnist
(train_images, traint_labels), (test_images, test_labels) = fmnist.load_data()

#Noramalize the pixels values
train_images = train_images/255.0
test_images = test_images/255.0

train_shape = train_images.shape
test_schape = test_images.shape

print(f'There are {train_shape[0]} Training examples with the shape ({train_shape[1], train_shape[2]})')
print(f'There are {test_schape[0]} Test examples with the shape ({test_schape[1], test_schape[2]})')

#Set the number of characters per row when printing
np.set_printoptions(linewidth=320)

index = 0
plt.imshow(train_images[index])
plt.show()
print(f'shape of the first image: {train_images[index].shape}')


#Callback accuracy 99%
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') >= 1):
            print("\n Reached 99% accuracy, so cancelling training!")
            self.model.stop_training = True

callbacks = myCallback()

#model
def build_Dense_model(callbacks):

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Hier you can use tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), if you did not specifie
    # the activation layer on the output layer.
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
    model.summary()
    print(f'\nDENSE MODEL TRAINING')
    history = model.fit(train_images, traint_labels, epochs=10, callbacks=[callbacks])
    return model, history

def build_CONV_model(callbacks):

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
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

def conv_and_pooling():
    f, axarr = plt.subplot(3, 4)

    FIRST_IMAGE = 0
    SECOND_IMAGE = 23
    THIRD_IMAGE = 28
    CONVOLUTION_NUMBER = 1

    layers_outputs = [layer.output for layer in model_Conv.layers]
    #functional api like
    activation_model = keras.models.Model(input=model_Conv.input, outputs=layers_outputs)

    for x in range(0, 4):
        f1 = activation_model.predict(test_images[FIRST_IMAGE].reshape(1, 28, 28, 1))[x]
        axarr[0, x].imshow(f1[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
        axarr[0, x].grid(False)

        f2 = activation_model.predict(test_images[SECOND_IMAGE].reshape(1, 28, 28, 1))[x]
        axarr[1, x].imshow(f2[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
        axarr[1, x].grid(False)

        f3 = activation_model.predict(test_images[THIRD_IMAGE].reshape(1, 28, 28, 1))[x]
        axarr[2, x].imshow(f3[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
        axarr[2, x].grid(False)




if __name__ == '__main__':

    model_Dense, history = build_Dense_model(callbacks)

    print(f'\nMODEL EVALUATION')
    dense_evaluation = model_Dense.evaluate(test_images, test_labels)
    print(f'EVALAUTION OF DENSE MODEL: {dense_evaluation}')

    model_Conv, history = build_CONV_model(callbacks)
    print(f'\nMODEL EVALUATION')
    dense_evaluation = model_Conv.evaluate(test_images, test_labels)
    print(f'EVALAUTION OF CONV MODEL: {dense_evaluation}')

    #conv_and_pooling()

