# Exploring Overfitting in NLP

import csv
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from collections import Counter

from tensorflow import keras

# Defining global variables
EMBEDDING_DIM = 100
MAXLEN = 16
TRUNCATING = 'post'
PADDING = 'post'
OOV_TOKEN = "<OOV>"
MAX_EXAMPLES = 160000  # 10% of total number of items in the original dataset
TRAINING_SPLIT = 0.9

sentiment_csv = './sentiment140/training.1600000.processed.noemoticon.csv'
with open(sentiment_csv, 'r') as csv_file:
    print(f'First data point looks like this: \n\n {csv_file.readline()}')
    print(f'Second data point looks like this: \n\n {csv_file.readline()}')

def parse_data_from_file(filename):
    """
    Extracts sentences and labels from a csv file
    :param filename: (string) path to the csv file
    :return: (list of string, list of string) tuple containing list of sentences and labels
    """
    sentences = []
    labels = []

    with open(filename, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        #next(reader)
        for line in reader:
            sentences.append(line[5])
            if line[0] == '4':
                labels.append(1)
            else:
                labels.append(0)
    return sentences, labels

def select_part_dataset(sentences, labels, MAX_EXAMPLES):
    # bundle the two lists
    sentences_labels = list(zip(sentences, labels))

    # Performing random sampling
    random.seed(42)
    sentences_labels = random.sample(sentences_labels, MAX_EXAMPLES)

    # Unpack back into two separated lst
    sentences, labels = zip(*sentences_labels)

    return sentences, labels

def train_val_split(sentences, labels, training_split):
    """
    Splits the data into training and validation sets
    :param sentences: (List of strings) lower-cased sentences without stopwords
    :param labels: (list of string) list of labels
    :param training_split: (float) proportion of the dataset to convert to includ in the train
    :return: train_sentences, validation_sentences, train_labels, validation_labels - lists containing the data splits
    """
    #Compute the number of sentences that will be used for training (should be an integer)
    train_size = round(training_split*len(sentences))

    # Split the sentences and labels into trainvalidation splits
    train_sentences = sentences[:train_size]
    train_labels = labels[:train_size]

    validation_sentences = sentences[train_size:]
    validation_labels = labels[train_size:]

    return train_sentences, train_labels, validation_sentences, validation_labels

def fit_tokenizer(train_sentences, oov_token):
    """
    Instantiates the Tokenizer class on the training sentences
    :param train_sentences: (list of srting) lower-cased sentences without stopwords used for training
    :param num_words: (int) number of words to keep when tokenizing
    :param oov_token: (string) - Symbol for the out-of-vocabulary token
    :return: tokenizer (object) an instance of the Tokenizer class containing the word index dictionary
    """
    # Initialize the Tokenizer class
    tokenizer = keras.preprocessing.text.Tokenizer(oov_token=oov_token)

    #Fit the tokenizer to the training sentences
    tokenizer.fit_on_texts(train_sentences)

    return tokenizer

def seq_and_pad(sentences, tokenizer, padding, maxlen, truncating):
    """
    Generates an array of token sequences and pads them to the same length
    :param sentences: (list of string): list of sentences to tekenize and pad
    :param tokenizer: (object): Tokenizer instance containing the word index dictionary
    :param padding: (string) type of padding to use
    :param maxlen: (int) maximum length of the token sequence
    :return: padded_sequences (array of int) tokenized sentences padded to the same length
    """
    # Convert sentences to sequences
    sequences = tokenizer.texts_to_sequences(sentences)

    # Pad the sequences using the correct padding and maxlen
    padded_sequences = keras.preprocessing.sequence.pad_sequences(sequences, maxlen=maxlen, padding=padding, truncating=truncating)

    return padded_sequences

def get_glove_embedding_vector(GLOVE_FILE):
    #Initialize an empty embeddings index dictionnary
    glove_embedding = {}

    # read file and fill glove_embedding with its contens
    with open(GLOVE_FILE, encoding="utf8") as file:
        for line in file:
            values = line.split()
            word = values[0]
            coefs = np.array(values[1:], dtype='float32')
            glove_embedding[word] = coefs
    return glove_embedding

def get_the_embedding_matrix(glove_embedding, word_index, EMBEDDING_DIM):
    # Initialize an empty array with the appropriate size
    embedding_matrix = np.zeros((len(word_index)+1, EMBEDDING_DIM))

    # Iterate all of the words in the vocabulary and if the vector representation for
    #each word exist within GloVe's representations, save it in the embeddings_matrix array
    for word, index in word_index.items():
        embedding_vector = glove_embedding.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector

    return embedding_matrix

def create_model(word_index, embedding_dim, maxlen, embedding_matrix, lstm_dim, dense_dim):
    """
    Create a binary sentiment classifier model
    :param word_index: (dictionary) Words from dataset an their index
    :param embedding_dim: (int) dimensionality of the Embedding layer output
    :param maxlen: (int) lenth of the input sequences
    :param embedding_matrix: (array) predefined weights of the embeddings
    :return: model (tf.keras model) the sentiment classifier model
    """
    model = keras.Sequential([
        keras.layers.Embedding(len(word_index)+1, embedding_dim, input_length=maxlen, weights=[embedding_matrix], trainable=False),
        keras.layers.Dropout(0.2),
        keras.layers.Bidirectional(keras.layers.LSTM(lstm_dim)),
        keras.layers.Dense(dense_dim, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.summary()
    model.compile(loss=keras.losses.binary_crossentropy, optimizer='adam', metrics=['accuracy'])

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




if __name__ == '__main__':
    sentences, labels = parse_data_from_file(sentiment_csv)
    print(f'dataset contains {len(sentences)} examples\n')
    print(f'Text of second example should look like this \n{sentences[1]}\n')
    print(f'Text of fourth example should look like this \n{sentences[3]}')
    print(f'\nLabels of last 5 examples sould look like this \n{labels[-20:]}')
    print(Counter(labels))

    sentences_, labels_ = select_part_dataset(sentences, labels, MAX_EXAMPLES)

    #Split Dataset
    train_sentences, train_labels, validation_sentences, validation_labels = train_val_split(sentences_, labels_, TRAINING_SPLIT)
    #Test the function
    print(f'There are {len(train_sentences)} sentences for training.\n')
    print(f'There are {len(train_labels)} labels for training.\n')

    print(f'There are {len(validation_sentences)} sentences for validation.\n')
    print(f'There are {len(validation_labels)} labels for validation.\n')

    #Tokenizer
    tokenizer = fit_tokenizer(train_sentences, oov_token=OOV_TOKEN)
    word_index = tokenizer.word_index

    # Test the funtion
    print(f'Vocabulary contains {len(word_index)} words\n')
    print('<oov> token included in vocabulary' if "<OOV>" in word_index else "<OOV> token NOT included in vocabulary")

    # Padding
    train_padded_sequences = seq_and_pad(train_sentences, tokenizer, padding=PADDING, maxlen=MAXLEN, truncating=TRUNCATING)
    val_padded_sequences = seq_and_pad(validation_sentences, tokenizer, padding=PADDING, maxlen=MAXLEN, truncating=TRUNCATING)

    # Test the function
    print(f'padded training sequences have shape: {train_padded_sequences.shape}')
    print(f'padded validation sequences have shape: {val_padded_sequences.shape}')

    train_labels = np.array(train_labels)
    validation_labels = np.array(validation_labels)

    #### Using Predefined Embeddings
    # This time we will not be learning embeddings from the dataset but we will using pre-trained word vectors .
    # In partucular we will be using the 100 dimension version of Glove from Stanford
    GLOVE_FILE = './GloVe100d/glove.6B.100d.txt'
    glove_embedding = get_glove_embedding_vector(GLOVE_FILE=GLOVE_FILE)

    #Lets take a look at the vector for the word dog
    test_word = 'dog'
    test_vector = glove_embedding[test_word]
    print(f'Vector representation of the word {test_vector} loos like this: \n\n{test_vector}')
    print(f'Each word vector has shape: {test_vector}')

    # Get the embedding_matrix of our words from word_index from GloVe Embedding
    embedding_matrix = get_the_embedding_matrix(glove_embedding=glove_embedding, word_index=word_index, EMBEDDING_DIM=EMBEDDING_DIM)

    # Build model
    embedding_dim = 100
    lstm_dim = 128
    dense_dim = 28
    NUM_EPOCHS = 10
    BATCH_SIZE = 64
    model = create_model(word_index=word_index, embedding_dim=embedding_dim, maxlen=MAXLEN,
                         embedding_matrix=embedding_matrix, lstm_dim=lstm_dim, dense_dim=dense_dim)
    # Train the model
    history = model.fit(train_padded_sequences, train_labels,
                             validation_data=(val_padded_sequences, validation_labels), epochs=NUM_EPOCHS)

    plot_function(history)