# CNN Data Augmentation on Horses or Humans Dataset

import os

import numpy as np
import wget

import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
import matplotlib.image as mping

current_dir = os.getcwd()
output_path = os.path.join(current_dir, 'Horse-or-human')

train_horse_dir = os.path.join('./horse-or-human/horses/train')
train_human_dir = os.path.join('./horse-or-human/humans/train')

validation_horse_dir = os.path.join('./horse-or-human/horses/validation')
validation_human_dir = os.path.join('./horse-or-human/humans/validation')

#Load sample train human igame and check the size
# Images Names
list_humans_images_names = os.listdir(train_human_dir)
# Fist image path
first_image_path = os.path.join(train_human_dir, list_humans_images_names[0])
# Load image
image = keras.utils.load_img(first_image_path)
#Check the size
print(f'the size of the first image in Humans train data is {image.size}')

#total number of hores and humans images in the directories
print(f'total training horses images: {len(os.listdir(train_horse_dir))}')
print(f'total training human images: {len(os.listdir(train_human_dir))}')
print(f'total validation horses images: {len(os.listdir(validation_horse_dir))}')
print(f'total validation human images: {len(os.listdir(validation_human_dir))}')

def data_augmentation():

    ###### TRAIN ########
    # Rescall all the images by 1/255.0
    train_data_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1/255,
                                                                  rotation_range=40,
                                                                  width_shift_range=0.2,
                                                                  height_shift_range=0.2,
                                                                  shear_range=0.2,
                                                                  zoom_range=0.2,
                                                                  horizontal_flip=True,
                                                                  fill_mode='nearest')
    # Flow training images in batches of 128 using ImageDataGenerator
    train_generator = train_data_gen.flow_from_directory(directory = './horse-or-human_/train',
                                                         target_size=(300, 300),
                                                         batch_size=128,
                                                         class_mode='binary')
    ###### VALIDATION ########
    # Rescall all the images by 1/255.0
    validation_data_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
    # Flow training images in batches of 128 using ImageDataGenerator
    validation_generator = validation_data_gen.flow_from_directory(directory = './horse-or-human_/validation',
                                                         target_size=(300, 300),
                                                         batch_size=32,
                                                         class_mode='binary')
    return train_generator, validation_generator

def build_CONV_model():

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),


        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    # Hier you can use tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), if you did not specifie
    # the activation layer on the output layer.
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=['accuracy'])
    model.summary()
    print(f'\nCONV MODEL TRAINING')
    return model

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') >= 0.995):
            print("\n Reached 99% accuracy, so cancelling training!")
            self.model.stop_training = True

def plot_function(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and validation accuracy')

    plt.figure()

    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')

    plt.legend()
    plt.show()

if __name__ == '__main__':
    #show_images()
    #build model
    model = build_CONV_model()
    #callacks
    callbacks = myCallback()
    #Train Generator
    train_generator, validation_generator = data_augmentation()
    #Fit model
    history = model.fit(train_generator,
              steps_per_epoch=8,
              epochs=20,
              verbose=1,
              validation_data=validation_generator,
              validation_steps=8)

    plot_function(history)
