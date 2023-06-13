################ Normalize Images #################

#Normalize the pixel values of train and test images
train_images = train_images / 255.0
test_images = test_images / 255.0

################################# Callback functions ##############################################
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

######################################### Build model with Image Generator ###########################
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

    ########################################### Predictions  ##################################################
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

#################################################### Plot loss ##################################################
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

############################# Load Zip File ##########################################
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

##################################### Download the pre-trained weights ###################################
url_preTrainedWeights = 'https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
filename_ = wget.download(url_preTrainedWeights, out='./inceptionv3')