# Building Models for the IMDB Reviews Dataset

from tensorflow import keras
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

#Download the dataset
imdb, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)

# Get the train and test splits
train_data, test_data = imdb['train'], imdb['test']

#Initialize sentences and labels
training_sentences = []
training_labels = []

testing_sentences = []
testing_labels = []

# Loop over all training examples and save the sentences and the labels
for sentence, label in train_data:
    training_sentences.append(sentence.numpy().decode('utf8'))
    training_labels.append(label.numpy())

# Loop over all testing examples and save the sentences and the labels
for sentence, label in test_data:
    testing_sentences.append(sentence.numpy().decode('utf8'))
    testing_labels.append(label.numpy())

# convert the label lists to numpy arrays
training_labels_final = np.array(training_labels)
testing_labels_final = np.array(testing_labels)

# Parameter
vocab_size = 10000
max_length = 120
trunc_type = 'post'
oov_tok = "<OOV>"

embedding_dim = 16
dense_dim = 6
NUM_EPOCHS = 10
BATCH_SIZE = 128

# Initialize The  Tokenizer Class
tokenizer = keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token=oov_tok)

# Generate the word_index dictionary for the training sentences
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index

#Generate and pad the training sequences
sequences = tokenizer.texts_to_sequences(training_sentences)
padded_sequences = keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_length, truncating=trunc_type)

#Generate and pad the test sequences
testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded_sequences = keras.preprocessing.sequence.pad_sequences(testing_sequences, maxlen=max_length)

model_flatten = keras.Sequential([
    keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    keras.layers.Flatten(),
    keras.layers.Dense(dense_dim, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

#model_flatten.summary()
model_flatten.compile(loss=keras.losses.binary_crossentropy, optimizer='adam', metrics=['accuracy'])

# Train the model
#history = model_flatten.fit(padded_sequences, training_labels_final, validation_data=(testing_padded_sequences, testing_labels_final), epochs=NUM_EPOCHS)

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

#plot_function(history)

############################### LSTM #######################################
embedding_dim = 16
lstm_dim = 32
dense_dim = 6
NUM_EPOCHS = 10
BATCH_SIZE = 128

model_lstm = keras.Sequential([
    keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    keras.layers.Bidirectional(keras.layers.LSTM(lstm_dim)),
    keras.layers.Dense(dense_dim, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model_lstm.summary()
model_lstm.compile(loss=keras.losses.binary_crossentropy, optimizer='adam', metrics=['accuracy'])

# Train the model
history = model_lstm.fit(padded_sequences, training_labels_final, validation_data=(testing_padded_sequences, testing_labels_final), epochs=NUM_EPOCHS)
plot_function(history)

############################### GRU ########################################
# The Gated Tedurrent Unit or GRU is usually referred to as a simpler version of LSTM
embedding_dim = 16
gru_dim = 32
dense_dim = 6
NUM_EPOCHS = 10
BATCH_SIZE = 128

model_gru = keras.Sequential([
    keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    keras.layers.Bidirectional(keras.layers.GRU(gru_dim)),
    keras.layers.Dense(dense_dim, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

#model_gru.summary()
model_gru.compile(loss=keras.losses.binary_crossentropy, optimizer='adam', metrics=['accuracy'])

# Train the model
#history = model_gru.fit(padded_sequences, training_labels_final, validation_data=(testing_padded_sequences, testing_labels_final), epochs=NUM_EPOCHS)
plot_function(history)


############################### CONVOLUTION ########################################
# The Gated Tedurrent Unit or GRU is usually referred to as a simpler version of LSTM
embedding_dim = 16
filters = 128
kernel_size = 5
dense_dim = 6
NUM_EPOCHS = 10
BATCH_SIZE = 128

model_conv = keras.Sequential([
    keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(dense_dim, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model_conv.summary()
model_conv.compile(loss=keras.losses.binary_crossentropy, optimizer='adam', metrics=['accuracy'])

# Train the model
history = model_conv.fit(padded_sequences, training_labels_final, validation_data=(testing_padded_sequences, testing_labels_final), epochs=NUM_EPOCHS)
plot_function(history)