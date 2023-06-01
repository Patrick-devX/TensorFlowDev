# TEXT CLASSIFICATION-MULTI Layer LSTM- DAtaset pretokenized IMDB Reviews dataset

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
features = 16
lstm_dim = 8

# Define array input with random values
random_input = np.random.rand(batch_size, timesteps, features)
print(f'shape of input array: {random_input.shape}')

#Define LSTM tha return a sequence
lstm_rs1 = keras.layers.LSTM(lstm_dim, return_sequences=False)
result_rsf = lstm_rs1(random_input)
print(f'shape of LSTM output with return_sequence False {result_rsf.shape}')

#Define LSTM tha return a sequence
lstm_rs2 = keras.layers.LSTM(lstm_dim, return_sequences=True)
result_rst = lstm_rs2(random_input)
print(f'shape of LSTM output with return_sequence True {result_rst.shape}')

##### Build and compile the model #####
#Hyperparameter
embedding_dim = 64
lstm1_dim = 64
lstm2_dim = 32
dense_dim = 64
NUM_EPOCHS = 10

model = keras.Sequential([
    keras.layers.Embedding(tokenizer.vocab_size, embedding_dim, input_length=BATCH_SIZE),
    keras.layers.Bidirectional(keras.layers.LSTM(lstm1_dim, return_sequences=True)),
    keras.layers.Bidirectional(keras.layers.LSTM(lstm1_dim)),
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