# Training a Sarcasm Detection Model Using Bidirectional LSTMs

import wget
import os
import io
import json

from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

def load_and_parse_dataset():

    url = 'https://storage.googleapis.com/tensorflow-1-public/course3/sarcasm.json'
    if not os.listdir('./sarcasm'):
        train_files = wget.download(url, out='./sarcasm')

    # Load the json File
    with open ('sarcasm/sarcasm.json', 'r') as f:
        datastore = json.load(f)

    print(f' The Headline is: {datastore[0]}')
    print(f'The sarcastic Headline is: {datastore[20000]}')

    # Collect data separately
    sentences = []
    labels = []
    urls = []

    for lines in datastore:
        sentences.append(lines['headline'])
        labels.append(lines['is_sarcastic'])
        urls.append(lines['article_link'])
    return sentences, labels

def split_dataset(sentences, labels, training_size):
    # Split the sentences
    training_sentences = sentences[0:training_size]
    testing_sentences = sentences[training_size:]

    # Split labes
    training_labels = labels[0:training_size]
    testing_labels = labels[training_size:]

    return (training_sentences, training_labels) , (testing_sentences, testing_labels)

def data_processing(training_sentences, testing_sentences, training_labels, testing_labels,
                    vocab_size, max_length, trunc_type, padding, oov_tok):
    # Initialize the Tokenizer Class
    tokenizer = keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token=oov_tok)

    # Generate the word index dictionary
    tokenizer.fit_on_texts(training_sentences)
    word_index = tokenizer.word_index

    #Generate and pad the training sequences
    training_sequences = tokenizer.texts_to_sequences(training_sentences)
    training_padded = keras.preprocessing.sequence.pad_sequences(training_sequences, maxlen=max_length, truncating=trunc_type, padding=padding)

    #Generate and pad testing sequences
    testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
    testing_padded = keras.preprocessing.sequence.pad_sequences(testing_sequences, maxlen=max_length, padding=padding, truncating=trunc_type)

    # Convert the labels lists into numpy arrays
    training_labels = np.array(training_labels)
    testing_labels = np.array(testing_labels)

    return training_padded, testing_padded, training_labels, testing_labels, word_index

def build_model(vocab_size, embedding_dim, lstm_dim, dense_dim, max_length):

    model_lstm = keras.Sequential([
        keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        keras.layers.Bidirectional(keras.layers.LSTM(lstm_dim)),
        keras.layers.Dense(dense_dim, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model_lstm.summary()
    model_lstm.compile(loss=keras.losses.binary_crossentropy, optimizer='adam', metrics=['accuracy'])

    return model_lstm

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

if __name__ == '__main__':

    sentences, labels = load_and_parse_dataset()

    # Split Dataset
    training_size = 20000
    (training_sentences, training_labels), (testing_sentences, testing_labels) = split_dataset(sentences, labels, training_size)

    vocab_size = 10000
    max_length = 120
    trunc_type = 'post'
    padding = 'post'
    oov_tok = '<OOV>'

    training_padded, testing_padded, training_labels, testing_labels, word_index = data_processing(training_sentences, testing_sentences,
                                                                                                   training_labels, testing_labels, vocab_size,
                                                                                                   max_length, trunc_type, padding, oov_tok)
    embedding_dim = 16
    lstm_dim = 32
    dense_dim = 6
    NUM_EPOCHS = 10
    BATCH_SIZE = 128

    model_lstm = build_model(vocab_size, embedding_dim, lstm_dim, dense_dim, max_length)
    # Train the model
    history = model_lstm.fit(training_padded, training_labels,
                             validation_data=(testing_padded, testing_labels), epochs=NUM_EPOCHS)
    plot_function(history)