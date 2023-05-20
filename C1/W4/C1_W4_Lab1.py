# Training with ImageDataGenerator

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

#See what the filenames look like in the horses and humans training directories
train_horse_names = os.listdir(train_horse_dir)
print(train_horse_names[:10])

train_human_names = os.listdir(train_human_dir)
print(train_human_names[:10])

nrows = 4
ncols = 4

train_human_names = os.listdir(train_human_dir)
print(train_human_names[:10])
train_horse_names = os.listdir(train_horse_dir)
print(train_horse_names[:10])

#total number of hores and humans images in the directories
print(f'total training horses images: {len(os.listdir(train_horse_dir))}')
print(f'total training human images: {len(os.listdir(train_human_dir))}')


def show_images():
    # Index for iterating over images
    image_index = 0

    fig = plt.gcf()
    fig.set_size_inches(ncols, nrows)

    image_index = image_index +8

    eight_human_pics = [os.path.join(train_human_dir, fname) for fname in train_human_names[image_index-8:image_index]]
    eight_horses_pics = [os.path.join(train_horse_dir, fname) for fname in train_horse_names[image_index-8:image_index]]

    for i, img_path in enumerate(eight_horses_pics + eight_human_pics):

        sp = plt.subplot(nrows, ncols, i+1)
        sp.axis('Off')
        img = mping.imread(img_path)
        plt.imshow(img)
        plt.show()

def data_augmentation():

    ###### TRAIN ########
    # Rescall all the images by 1/255.0
    train_data_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
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

if __name__ == '__main__':
    #show_images()
    #build model
    model = build_CONV_model()
    #callacks
    callbacks = myCallback()
    #Train Generator
    train_generator, validation_generator = data_augmentation()
    #Fit model
    model.fit(train_generator, steps_per_epoch=8, epochs=17, verbose=1,
              validation_data=validation_generator, validation_steps=8)

    ############# Predictions ##############
    test_human_names = os.listdir('./test_images/')
    test_images = [os.path.join('./test_images/', fname)
                   for fname in test_human_names]

    for myTestImage in test_images:
        img = keras.utils.load_img(myTestImage, target_size=(300, 300))
        img_array = keras.utils.img_to_array(img)
        img_array/=255
        img_array = np.expand_dims(img_array, axis=0)
        images = np.vstack([img_array])
        classes = model.predict(images)
        print(classes[0])
        if classes[0]>0.5:
            print(myTestImage + ' is a human')
        else:
            print(myTestImage + ' is a horse')
