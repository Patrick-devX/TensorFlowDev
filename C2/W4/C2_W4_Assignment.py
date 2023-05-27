# Multi-Class Classification

import os
import string

import keras.utils
import matplotlib.pyplot as plt
import wget
import gdown
import csv

import numpy as np
from tensorflow import keras
import visualkeras

url_train = 'https://drive.google.com/uc?id=1z0DkA9BytlLxO1C0BAWzknLyQmZAp0HR'
url_validation = 'https://drive.google.com/uc?id=1z1BIj4qmri59GWBG4ivMNFtpZ4AXIbzg'


if not os.listdir('./sign-language-mnist/train') or not os.listdir('./sign-language-mnist/validation'):
    train_files = gdown.download(url_train, output='./sign-language-mnist/train/sign_mnist_train.csv')
    validation_files = gdown.download(url_validation, output='./sign-language-mnist/validation/sign_mnist_test.csv')

#Base directories
TRAINING_FILE = './sign-language-mnist/train/sign_mnist_train.csv'
VALIDATION_FILE = './sign-language-mnist/validation/sign_mnist_test.csv'

#Unlike previous labs, there are no actual images provided. Instead we get the data serialized as csv files
# We take a look within the file

with open(TRAINING_FILE) as train_file:
    line = train_file.readline()
    #print(f'First Line (header) looks like this:\n{line}')
    line = train_file.readline()
    #print(f'Each subsequent Line (data points) looks like this:\n{line}')


# GRADED FUNCTION: parse_data_from_input
def parse_data_from_input(filename):
    with open(filename) as file:
        csv_reader = csv.reader(file, delimiter=',')
        labels = []
        images = []
        # Skip header
        next(csv_reader, None)
        for row in csv_reader:
            label = row[0]
            image = row[1:]
            image = np.reshape(image, (28, 28))
            labels.append(label)
            images.append(image)

        labels = np.array(labels).astype('float')
        images = np.array(images).astype('float')
        return images, labels

def plot_categories(training_images, training_labels):
    fig, axes = plt.subplot(1, 10, figsize=(16, 15))
    axes = axes.flatten()
    letters = list(string.ascii_lowercase)

    for k in range(10):
        img = training_images[k]
        img = np.expand_dims(img, axis=-1)
        img = keras.utils.array_to_img(img)
        ax = axes[k]
        ax.imshow(img, cmap='Greys_r')
        ax.set_title(f'{letters[int(training_labels[k])]}')
        ax.set_axis_off()
    plt.tight_layout()
    plt.show()


def train_val_generators(training_images, training_labels, validation_images, validation_labels):
    training_images = np.expand_dims(training_images, axis=3)
    validation_images = np.expand_dims(validation_images, axis=3)

    # Instantiate the ImageDataGenerator class and also we need to normalize pixel values and set arguments to augment the images (if desired)
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_generator = train_datagen.flow(x=training_images,
                                         y=training_labels,
                                         batch_size=260)

    validation_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1 / 255)

    validation_generator = validation_datagen.flow(x=validation_images,
                                                   y=validation_labels,
                                                   batch_size=260)

    return train_generator, validation_generator


def create_model():
    model = keras.models.Sequential([
        keras.layers.Conv2D(352, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Conv2D(640, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Conv2D(640, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(2, 2),


        keras.layers.Flatten(),
        keras.layers.Dense(1056, activation='relu'),
        keras.layers.Dense(26, activation='softmax')])

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.summary()
    return model

def plot_function(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.grid(True)
    plt.legend()
    plt.figure()

    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    training_images, training_labels = parse_data_from_input(TRAINING_FILE)
    validation_images, validation_labels = parse_data_from_input(VALIDATION_FILE)

    # Print shape
    print(f'Training images has shape: {training_images.shape} and type: {training_images.dtype}')
    print(f'Training labels has shape: {training_labels.shape} and type: {training_labels.dtype}')

    print(f'Validation images has shape: {validation_images.shape} and type: {validation_images.dtype}')
    print(f'Validation labels has shape: {validation_labels.shape} and type: {validation_labels.dtype}')

    #plot_categories(training_images, training_labels)

    train_generator, validation_generator = train_val_generators(training_images, training_labels, validation_images, validation_labels)
    # Print gen shape
    print(f'Images of training generators has shape: {train_generator.x.shape}')
    print(f'Labels of training generators has shape: {train_generator.y.shape}')

    print(f'Images of validation generators has shape: {validation_generator.x.shape}')
    print(f'Labels of validation generators has shape: {validation_generator.y.shape}')

    model = create_model()

    history = model.fit(train_generator,
                        steps_per_epoch=100,
                        epochs=15,
                        verbose=1,
                        validation_data=validation_generator)

    plot_function(history)

    model_img_file = './model_architecture/model.png'
    #keras.utils.plot_model(model, to_file=model_img_file,
                           #show_dtype=True,
                           #show_layer_activations=True,
                           #show_shapes=True,
                           #show_layer_names=True)
    visualkeras.layered_view(model, legend=True)

    # Model Evaluation
    print(f'The accuracy of the model is: {model.evaluate(validation_images, validation_labels)[1]*100} %')