# Transfert Learning

import wget
import zipfile
import os

import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt


url_train = 'https://storage.googleapis.com/tensorflow-1-public/course2/week3/horse-or-human.zip'
url_validation = 'https://storage.googleapis.com/tensorflow-1-public/course2/week3/validation-horse-or-human.zip'


if not os.listdir('./horse-or-human/train') or not os.listdir('./horse-or-human/validation'):
    train_files = wget.download(url_train, out='./horse-or-human/train')
    validation_files = wget.download(url_validation, out='./horse-or-human/validation')

    #train
    zip_ref = zipfile.ZipFile('./horse-or-human/train/horse-or-human.zip')
    zip_ref.extractall(os.path.join('./horse-or-human', 'tmp/training'))
    zip_ref.close()

    #validation
    zip_ref = zipfile.ZipFile('./horse-or-human/validation/validation-horse-or-human.zip')
    zip_ref.extractall(os.path.join('./horse-or-human', 'tmp/validation'))
    zip_ref.close()

# Define the training and validation bas directories
train_dir = './horse-or-human/tmp/training'
validation_dir = './horse-or-human/tmp/validation'

# Directory with training forses images
train_horses_dir = os.path.join(train_dir, 'horses')
# Directory with training humans images
train_humans_dir = os.path.join(train_dir, 'humans')

# Directory with validation forses images
validation_horses_dir = os.path.join(validation_dir, 'horses')
# Directory with validation humans images
validation_humans_dir = os.path.join(validation_dir, 'humans')

#Check the number of images for each classes
print(f'there are {len(os.listdir(train_horses_dir))} imgages of horses for training. \n')
print(f'there are {len(os.listdir(train_humans_dir))} imgages of humans for training. \n')

print(f'there are {len(os.listdir(validation_horses_dir))} imgages of horses for validation. \n')
print(f'there are {len(os.listdir(validation_humans_dir))} imgages of humans for validation. \n')

#Take a look to sample images
print('Sample horse image')
plt.imshow(keras.utils.load_img(f'{os.path.join(train_horses_dir, os.listdir(train_horses_dir)[0])}'))
plt.show()

print('Sample human image')
plt.imshow(keras.utils.load_img(f'{os.path.join(train_humans_dir, os.listdir(train_humans_dir)[0])}'))
plt.show()

#Check the Image resolution ex. horse image
sample_image = keras.utils.load_img(f'{os.path.join(train_horses_dir, os.listdir(train_horses_dir)[0])}')
# Convert the image into its numpy array representation
sample_array = keras.utils.img_to_array(sample_image)
print(f'Each image has shape: {sample_array.shape}')

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
                                                         target_size=(300, 300),
                                                         batch_size=20,
                                                         class_mode='binary')
    ###### VALIDATION ########
    # Rescall all the images by 1/255.0
    validation_data_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255)
    # Flow training images in batches of 128 using ImageDataGenerator
    validation_generator = validation_data_gen.flow_from_directory(directory=validation_dir,
                                                                   target_size=(300, 300),
                                                                   batch_size=20,
                                                                   class_mode='binary')
    return train_generator, validation_generator

def create_pre_trained_model(local_weights_file):
    """
    Initializes an InceptionV3 model.
    :param local_weights_file: (String) path pointing to a pretrained weights H5 file
    :return: the initialized InceptionV3 model
    """
    # Set the input shape and remove the dense Layers.
    pre_trained_model = keras.applications.inception_v3.InceptionV3(input_shape=(300, 300, 3),
                                                                    include_top=False,
                                                                    weights=None)
    # load the downloaded pre trained weights
    pre_trained_model.load_weights(local_weights_file)

    # Freeze the weights of the layers
    for layer in pre_trained_model.layers:
        layer.trainable = False

    return pre_trained_model


class myCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') >= 0.999):
            print("\n Reached 99,9% accuracy, so cancelling training!")
            self.model.stop_training = True

#Pipelining the pretrained model wiht the own model

def output_of_last_layer(pre_trained_model):
    """
    Gets tha last layer output of a model
    :param pre_trained_model:
    :return: last_output: output of the model's last layer
    """
    last_layer = pre_trained_model.get_layer('mixed7')
    print(f'last layer output shape: {last_layer.output_shape}')
    last_output = last_layer.output
    return last_output

def create_final_model(pre_trained_model, last_layer_output):
    """
    Appends a custom model to the pretrained model
    :param pre_trained_model: (tf.keras.model) model that will accept the traintest inputs
    :param last_layer_output: last layer output of the pretrained model
    :return: combined model
    """
    x = keras.layers.Flatten()(last_layer_output)
    x = keras.layers.Dense(1024, activation='relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(pre_trained_model.input, outputs)
    model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
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
    plt.figure()

    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == '__main__':

    train_generator, validation_generator = train_validation_generators(train_dir, validation_dir)

    #### Create the pre trained model ###
    # Download the pre-trained weights
    url_preTrainedWeights = 'https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
    inception_ = wget.download(url_preTrainedWeights, out='./inceptionv3')

    # get the downloaded weights as variable
    local_weights_file = './inceptionv3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

    pre_trained_model = create_pre_trained_model(local_weights_file)
    pre_trained_model.summary()

    total_params = pre_trained_model.count_params()
    num_trainable_params = sum([w.shape.num_elements() for w in pre_trained_model.trainable_weights])
    print(f'there are {total_params:,} total parameters in this model.')
    print(f'there are {num_trainable_params:,} trainable parameters in this model.')

    #Output of last layer
    last_layer_output = output_of_last_layer(pre_trained_model)

    # Create final model
    model = create_final_model(pre_trained_model, last_layer_output)

    #Instpects parameters
    total_params = model.count_params()
    num_trainable_params = sum([w.shape.num_elements() for w in model.trainable_weights])
    print(f'there are {total_params:,} total parameters in this model.')
    print(f'there are {num_trainable_params:,} trainable parameters in this model.')

    callbacks = myCallback()
    # Fit model
    history = model.fit(train_generator,
                        steps_per_epoch=100,
                        epochs=20,
                        verbose=1,
                        validation_data=validation_generator,
                        callbacks=callbacks)

    plot_function(history)