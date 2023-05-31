# Training a binary classifier with the Sarcasm Dataset

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

def data_processing(sentences_train, sentences_test, padding_type, trunc_type, oov_tok, vocab_size):

    # Initialize tockenizer
    tokenizer = keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token=oov_tok)

    # Generate the word index dictionary
    tokenizer.fit_on_texts(sentences_train)
    word_index = tokenizer.word_index

    # Generate and pad the sequences
    training_sequences = tokenizer.texts_to_sequences(sentences_train)
    padded_sequences_train = keras.preprocessing.sequence.pad_sequences(training_sequences, maxlen=max_len, truncating=trunc_type, padding=padding_type)

    testing_sequences = tokenizer.texts_to_sequences(sentences_test)
    padded_sequences_test = keras.preprocessing.sequence.pad_sequences(testing_sequences, maxlen=max_len, truncating=trunc_type, padding=padding_type)

    return padded_sequences_train, padded_sequences_test,  tokenizer

def build_model():

    model = keras.Sequential([
        keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_len),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(24, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

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
    plt.legend()
    plt.figure()

    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.grid(True)
    plt.legend()
    plt.show()

def visualize_wordEmbedding(model, tokenizer_train):
    """
    Visualize the trained weights in the Embedding layer to see words that are clustered together.
    :return:
    """
    #Get embedding layer from the model
    embedding_layer = model.layers[0]
    # Embedding Weights
    embedding_weights = embedding_layer.get_weights()[0]

    # Print the shape. expected is (vocab_size, embedding_dim)
    print(embedding_weights.shape)

    #We need to generate two files
    #   vecs.tsv -contains the vector weights of each word in the vocabulary
    #    meta.tsv -contains the words in the vocabulary

    #GEt the index-word dictionary
    reverse_word_index = tokenizer_train.index_word

    # Open writable files
    out_v = io.open('./tensorboard/vecs.tsv', 'w', encoding='utf-8')
    out_m = io.open('./tensorboard/meta.tsv', 'w', encoding='utf-8')

    for word_num in range(1, vocab_size):
        # Get the word associated at the current index
        word_name = reverse_word_index[word_num]
        # Get the Emedding weights associated with the current index
        word_embedding = embedding_weights[word_num]
        #write the word name to out_m
        out_m.write(word_name + "\n")
        # Whrite the word embedding
        out_v.write("\t".join([str(x) for x in word_embedding]) + "\n")
    out_v.close()
    out_m.close()


if __name__ == '__main__':

    sentences, labels = load_and_parse_dataset()

    ##### Hyperparameters #####
    training_size = 20000
    vocab_size = 1000
    max_len = 32
    embedding_dim = 128

    ###### Split the Dataset #####
    training_sentences = sentences[:training_size]
    testing_sentences = sentences[training_size:]

    training_labels = labels[:training_size]
    testing_labels = labels[training_size:]

    ##### Parameters for Padding #####
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<oov>"

    padded_sequences_train, padded_sequences_test, tokenizer = data_processing(training_sentences, testing_sentences, padding_type, trunc_type, oov_tok, vocab_size)


    # Convert the labels lists into numpy arrays
    training_labels_final = np.array(training_labels)
    testing_labels_final = np.array(testing_labels)

    # Model parameter
    num_epochs = 30
    model = build_model()
    history = model.fit(padded_sequences_train, training_labels_final, epochs=num_epochs, validation_data=(padded_sequences_test, testing_labels_final), verbose=2)

    # Plots Loss and accuracy
    plot_function(history)

    #Visualize Word Embedding
    visualize_wordEmbedding(model, tokenizer)

