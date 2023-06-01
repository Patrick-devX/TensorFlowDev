# TEXT CLASSIFICATION-CONV1D Layer- DAtaset pretokenized IMDB Reviews dataset

import tensorflow_datasets as tfds
from tensorflow import keras

import matplotlib.pyplot as plt
import numpy as np

#Download the subword encoded pretokenized dataset
dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)

#Get the Tokenizer
tokenizer = info.features['text'].encoder

#### Prepare the dataset #####
BUFFER_SIZE = 10000
BATCH_SIZE = 256

# Get the train and test splits
train_data, test_data = dataset['train'], dataset['test']

# shuffle the training data
train_dataset = train_data.shuffle(BUFFER_SIZE)

# Batch and pad the dataset to the maximum length of the sequences
train_dataset = train_dataset.padded_batch(BATCH_SIZE)
test_dataset = test_data.padded_batch(BATCH_SIZE)

batch_size = 1
timesteps = 20
features = 20
filters = 128
lstm_dim = 8
kernel_size = 5

# Define array input with random values
random_input = np.random.rand(batch_size, timesteps, features)
print(f'shape of input array: {random_input.shape}')
print(random_input)

#Define LSTM tha return a sequence
conv1D = keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu')
result_conv1D = conv1D(random_input)
print(f'shape of conv1D output with return_sequence False {result_conv1D.shape}')

#Define LSTM tha return a sequence
gmp = keras.layers.GlobalMaxPooling1D()
result_gmp = gmp(random_input)
print(f'shape of globalmaxpooling1D output with return_sequence True {result_gmp.shape}')

##### Build and compile the model #####
#Hyperparameter
embedding_dim = 64
filters = 128
kernel_size = 5
dense_dim = 64
NUM_EPOCHS = 10

model = keras.Sequential([
    keras.layers.Embedding(tokenizer.vocab_size, embedding_dim),
    keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'),
    keras.layers.GlobalMaxPooling1D(),
    keras.layers.Dense(dense_dim, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.summary()
model.compile(loss=keras.losses.binary_crossentropy, optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(train_dataset, validation_data=test_dataset, epochs=NUM_EPOCHS)

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

plot_function(history)