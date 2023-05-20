 # Implementating Callbacks in TensorFlow using the MNIST Dataset
import os
import tensorflow as tf
from tensorflow import keras

# Get current directory
#current_dir = os.getcwd()
current_dir = r'C:\Users\tchuentep\OneDrive - Josera GmbH & Co. KG\Desktop\tensorFlow\TFDevCert'

#Append data/mnist.npz to the previous path to get the full path
data_path = os.path.join(current_dir, 'data')
data_path = os.path.join(data_path, 'mnist.npz')

#root_logdir = os.path.join(os.curdir, '../my_logs')

# Load Data
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()

#Shape
data_shape = x_train.shape
print(f'There are {data_shape[0]} examples with the shape ({data_shape[1], data_shape[2]})')

#Normalize pixel values
x_train = x_train/255.0

#Callback accuracy 99%
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') >= 0.99):
            print("\n Reached 99% accuracy, so cancelling training!")
            self.model.stop_training = True

callbacks = myCallback()

#model
def build_model():

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Hier you can use tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), if you did not specifie
    # the activation layer on the output layer.
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
    model.summary()
    history = model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])
    return model, history

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') >= 0.995):
            print("\n Reached 99% accuracy, so cancelling training!")
            self.model.stop_training = True

    callbacks = myCallback()

if __name__ == '__main__':

    model, history = build_model()
