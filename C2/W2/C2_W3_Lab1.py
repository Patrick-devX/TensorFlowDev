# Transfert Learning
# Prepare the dataset
import wget
import zipfile
import os

import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt

url = 'https://storage.googleapis.com/tensorflow-1-public/course2/cats_and_dogs_filtered.zip'
if not os.listdir('./cats_and_dogs'):
    filename = wget.download(url, out='./cats_and_dogs')

    zip_ref = zipfile.ZipFile('./cats_and_dogs/cats_and_dogs_filtered.zip')
    zip_ref.extractall(os.path.join('./cats_and_dogs', 'tmp/'))
    zip_ref.close()

base_dir = os.path.join('./cats_and_dogs/tmp', 'cats_and_dogs_filtered')

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')

validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

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
train_generator = train_data_gen.flow_from_directory(directory=train_dir,
                                                     target_size=(150, 150),
                                                     batch_size=20,
                                                     class_mode='binary')
###### VALIDATION ########
# Rescall all the images by 1/255.0
validation_data_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
# Flow training images in batches of 128 using ImageDataGenerator
validation_generator = validation_data_gen.flow_from_directory(directory=validation_dir,
                                                     target_size=(150, 150),
                                                     batch_size=20,
                                                     class_mode='binary')

#Download the pre-trained weights
url_preTrainedWeights = 'https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
filename_ = wget.download(url_preTrainedWeights, out='./inceptionv3')

#Load sample train human igame and check the size to know what input size we will give in the pre-trained model
# Images Names
list_dogs_images_names = os.listdir(train_dogs_dir)
# Fist image path
first_image_path = os.path.join(train_dogs_dir, list_dogs_images_names[0])
# Load image
image = keras.utils.load_img(first_image_path)
#image size
img_size = image.size
print(f'the size of the first image in Humans train data is {img_size}')

#get the downloaded weights as variable
local_weights_file = './inceptionv3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
#Initialize the base model
# Set the input shape and remove the dense Layers.
pre_trained_model = keras.applications.inception_v3.InceptionV3(input_shape=(150, 150, 3),
                                                                include_top=False,
                                                                weights=None)
#load the downloaded pre trained weights
pre_trained_model.load_weights(local_weights_file)

#Freeze the weights of the layers
for layer in pre_trained_model.layers:
    layer.trainable = False

pre_trained_model.summary()

# Choose 'mixed7' as the last layer of the bas model
last_layer = pre_trained_model.get_layer('mixed7')
print(f'last layer output shape: {last_layer.output_shape}')
last_output = last_layer.output

# Add Dense layer to the classifier

# Flatten the output layer to 1 dimension
x = keras.layers.Flatten()(last_output)
# add Fully connected layer with 1024 hiden units and ReLu activation
x = keras.layers.Dense(1024, activation='relu')(x)
# Add dropout
c = keras.layers.Dropout(0.2)(x)
# Add final sigmoid layer for binare classification
output = keras.layers.Dense(1, activation='sigmoid')(x)
#Append the dense network to the bas model
model = keras.Model(pre_trained_model.input, output)
model.summary()
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

#Fit model
history = model.fit(train_generator,
          steps_per_epoch=100,
          epochs=20,
          verbose=1,
          validation_data=validation_generator,
          validation_steps=50)

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
    plt.figure()

    plt.plot(epochs, loss, 'r', label='Training loss', grid=)
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.grid(True)
    plt.legend()
    plt.show()

plot_function(history)