# Multi-Class Classifier

import wget
import zipfile
import os

import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt

url_train = 'https://storage.googleapis.com/tensorflow-1-public/course2/week4/rps.zip'
url_validation = 'https://storage.googleapis.com/tensorflow-1-public/course2/week4/rps-test-set.zip'


if not os.listdir('./rock-paper-scissors/train') or not os.listdir('./rock-paper-scissors/validation'):
    train_files = wget.download(url_train, out='./rock-paper-scissors/train')
    validation_files = wget.download(url_validation, out='./rock-paper-scissors/validation')

    #train
    zip_ref = zipfile.ZipFile('./rock-paper-scissors/train/rps.zip')
    zip_ref.extractall(os.path.join('./rock-paper-scissors', 'tmp/training'))
    zip_ref.close()

    #validation
    zip_ref = zipfile.ZipFile('./rock-paper-scissors/validation/rps-test-set.zip')
    zip_ref.extractall(os.path.join('./rock-paper-scissors', 'tmp/validation'))
    zip_ref.close()

base_dir = './rock-paper-scissors/tmp/training/rps'

train_dir = './rock-paper-scissors/tmp/training/rps'
validation_dir = './rock-paper-scissors/tmp/validation/rps-test-set'

train_rock_dir = os.path.join(base_dir, 'rock')
train_paper_dir = os.path.join(base_dir, 'paper')
train_scissors_dir = os.path.join(base_dir, 'scissors')

print(f'total training rock images: {len(os.listdir(train_rock_dir))}')
print(f'total training paper images: {len(os.listdir(train_paper_dir))}')
print(f'total training scissors images: {len(os.listdir(train_scissors_dir))}')

def build_model():

    model = keras.models.Sequential([
        keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)),
        keras.layers.MaxPooling2D(2,2),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Conv2D(128, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Conv2D(128, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Flatten(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    model.summary()
    return model

# Prepare the ImageDataGenerator
def train_validation_generators (train_dir, validation_dir):
    """
    creates the training and validation data generators
    :param train_dir: (string) directory path containing the training images
    :param validation_dir: (string) directory path containing the validation images
    :return: training_generator, validation_generator
    """
    ###### TRAIN ########
    # Rescall all the images by 1/255.0
    train_data_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255,
                                                                  rotation_range=40,
                                                                  width_shift_range=0.2,
                                                                  height_shift_range=0.2,
                                                                  shear_range=0.2,
                                                                  zoom_range=0.2,
                                                                  horizontal_flip=True,
                                                                  fill_mode='nearest')
    # Flow training images in batches of 128 using ImageDataGenerator
    train_generator = train_data_gen.flow_from_directory(directory=train_dir,
                                                         target_size=(150, 150),
                                                         batch_size=20,
                                                         class_mode='categorical')
    ###### VALIDATION ########
    # Rescall all the images by 1/255.0
    validation_data_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255)
    # Flow training images in batches of 128 using ImageDataGenerator
    validation_generator = validation_data_gen.flow_from_directory(directory=validation_dir,
                                                                   target_size=(150, 150),
                                                                   batch_size=20,
                                                                   class_mode='categorical')
    return train_generator, validation_generator

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

class myCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') >= 0.999):
            print("\n Reached 99,9% accuracy, so cancelling training!")
            self.model.stop_training = True


if __name__ == '__main__':

    train_generator, validation_generator = train_validation_generators(train_dir, validation_dir)

    model = build_model()
    callbacks = myCallback()
    history = model.fit(train_generator,
                        steps_per_epoch=100,
                        epochs=17,
                        verbose=1,
                        validation_data=validation_generator,
                        callbacks=callbacks)

    plot_function(history)

    ############# Predictions ##############
    import numpy as np
    test_human_names = os.listdir('./test-images/')
    test_images = [os.path.join('./test-images/', fname) for fname in test_human_names]

    for myTestImage in test_images:
        img = keras.utils.load_img(myTestImage, target_size=(150, 150))
        img_array = keras.utils.img_to_array(img)
        img_array /= 255
        img_array = np.expand_dims(img_array, axis=0)
        images = np.vstack([img_array])
        classes = model.predict(images)


