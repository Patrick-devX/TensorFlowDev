# Using more sophisticated images with Convolutional Neural Networks

# Download and Inspect the Dataset

import os

import tensorflow as tf
from tensorflow import keras

train_dir = os.path.join(os.getcwd(), './dog-vs-cats/train')
validation_dir = os.path.join(os.getcwd(), './dog-vs-cats/validation')

train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')

validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

# See what files names look like
train_cat_fnames = os.listdir(train_cats_dir)
train_dog_fnames = os.listdir(train_dogs_dir)

# print number of training and validation images
print(f'total training dog images :', len(os.listdir(train_dogs_dir)))
print(f'total training cat images :', len(os.listdir(train_cats_dir)))

print(f'total validation dog images :', len(os.listdir(validation_dogs_dir)))
print(f'total validation cat images :', len(os.listdir(validation_cats_dir)))

def data_augmentation():

    ###### TRAIN ########
    # Rescall all the images by 1/255.0
    train_data_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
    # Flow training images in batches of 128 using ImageDataGenerator
    train_generator = train_data_gen.flow_from_directory(directory = './dog-vs-cats/train',
                                                         target_size=(150, 150),
                                                         batch_size=20,
                                                         class_mode='binary')
    ###### VALIDATION ########
    # Rescall all the images by 1/255.0
    validation_data_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
    # Flow training images in batches of 128 using ImageDataGenerator
    validation_generator = validation_data_gen.flow_from_directory(directory = './dog-vs-cats/validation',
                                                         target_size=(150, 150),
                                                         batch_size=20,
                                                         class_mode='binary')
    return train_generator, validation_generator


def build_CONV_model():

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
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
        tf.keras.layers.Dense(512, activation='relu'),
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

    # Fit model
    model.fit(train_generator, steps_per_epoch=8, epochs=30, verbose=1,
              validation_data=validation_generator, validation_steps=8)