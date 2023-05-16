import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# The MNIST Dataset is a collection of grayscale 28x28 pixel cothing images.
#Each Image is assiciated with the label as shown in this table.

label_name ={
    0: "T-shirttop",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot"
}

# Load Dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


print(train_images.shape)
print(train_labels.shape)

index = 0

#Set the number of characters per row when printing
np.set_printoptions(linewidth=320)
print(f'Label: {train_labels[index]}')
print(f'Label Name: {label_name[train_labels[index]]}')
print(f'\nImage Pixel Array:\n{train_images[index]}')
plt.imshow(train_images[index])
plt.show()

#Normalize the pixel values of train and test images
train_images = train_images / 255.0
test_images = test_images / 255.0


# How the activation function softmax works
def softmax_demo():
    inputs = np.array([[1.0, 3.0, 4.0, 2.0]])
    inputs = tf.convert_to_tensor(inputs)
    print(f'input to softmaxfunction: {inputs.numpy()}')
    # Feed the input to softmax function
    outputs = keras.activations.softmax(inputs)
    print(f'Output to softmaxfunction: {outputs.numpy()}')
    # Get the sum of all values after the softmax function
    sum_input = tf.reduce_sum(inputs)
    sum_output = tf.reduce_sum(outputs)

    return outputs, sum_output, sum_input

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') >= 0.9):
            print("\n Reached 80% accuracy, so cancelling training!")
            self.model.stop_training = True

class myCallback_loss(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('loss') <= 0.3):
            print("\n Loss is lower than 0.3 accuracy, so cancelling training!")
            self.model.stop_training = True

callbacks = myCallback()
callbacks_loss = myCallback_loss()


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
    model.fit(train_images, train_labels, epochs=10, callbacks=[callbacks_loss])
    return model


if __name__ == '__main__':
    #For a multiclass classifier, the sigmoid activation function has the drawback that the sum of its outputs won’t necessarily equal 1.
    #For that reason, practitioners often use softmax instead.Like sigmoid, softmax ‘squashes’ the output to values that range between 0 and 1,
    #but it also ensures that the total of all outputs sum to 1
    outputs, sum_output, sum_input = softmax_demo()
    model = build_model()
    evaluation = model.evaluate(test_images, test_labels, verbose=2)
    print(f'evaluation: {evaluation}')

    classifications = model.predict(test_images)
    print(f'predicted value: {classifications[0]}')
    # Print test_label
    print(f'is label: {test_labels[0]}')
    print(f'predicted label: {np.argmax(classifications[0])}')
    print(f'predicted label name: {label_name[np.argmax(classifications[0])]}')