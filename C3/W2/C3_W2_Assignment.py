# Diving deeper into the BBC News archive
# The BBC News Classification Dataset contains 2225 examples of news articles with their respective labels

import csv
import io
from tensorflow import keras
import numpy as np
import tensorflow
import matplotlib.pyplot as plt

with open('./bbc-text/bbc-text.csv', 'r') as csv_file:
    print(f'First line (header) looks like this: \n\n {csv_file.readline()}')
    print(f'Each data point looks like this: \n\n {csv_file.readline()}')


NUM_WORDS = 1000
EMBEDDING_DIM = 16
MAXLEN = 120
PADDING = 'post'
OOV_TOKEN = "<OOV>"
TRAINING_SPLIT = 0.8

def remove_stopwords(sentence):
    """
    remove a list of stop words
    :param sentence: sentence to remove the stopwords from
    :return: (string): lowcase sentence without the stopwords
    """
    # List of stop words
    stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
                 "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did",
                 "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have",
                 "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself",
                 "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's",
                 "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only",
                 "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd",
                 "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs",
                 "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
                 "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we",
                 "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's",
                 "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll",
                 "you're", "you've", "your", "yours", "yourself", "yourselves"]

    sentence = sentence.lower()

    sentence = ' '.join([word for word in sentence.split() if word not in stopwords])

    return sentence

def parse_data_from_file(filname):
    """
    Extracts sentences and labels from CSV file
    :param filname: (string).path to the csv file
    :return: sentences, labels (list of string, list of string): tuple containing lists of sentences and labels
    """

    sentences = []
    labels = []

    with open(filname, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        next(reader)
        for line in reader:
            labels.append(line[0])
            sentences.append(remove_stopwords(line[1]))

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

def fit_tokenizer(train_sentences, num_words, oov_token):
    """
    Instantiates the Tokenizer class on the training sentences
    :param train_sentences: (list of srting) lower-cased sentences without stopwords used for training
    :param num_words: (int) number of words to keep when tokenizing
    :param oov_token: (string) - Symbol for the out-of-vocabulary token
    :return: tokenizer (object) an instance of the Tokenizer class containing the word index dictionary
    """
    # Initialize the Tokenizer class
    tokenizer = keras.preprocessing.text.Tokenizer(num_words=num_words, oov_token=oov_token)

    #Fit the tokenizer to the training sentences
    tokenizer.fit_on_texts(train_sentences)

    return tokenizer

def seq_and_pad(sentences, tokenizer, padding, maxlen):
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
    padded_sequences = keras.preprocessing.sequence.pad_sequences(sequences, maxlen=maxlen, padding=padding)

    return padded_sequences

def tokenize_labels(all_labels, split_labels):
    """
    Tokenizes the labels
    :param all_labels: (list of strings) labels to generate the word_index from
    :param split_labels: (list of string) labels to tokenize
    :return: label_sequences (array of int) tokenized labes
    """
    # Instantiate the Tokenzer
    tokenizer = keras.preprocessing.text.Tokenizer()

    # Fit the tokenizer on all the labels
    tokenizer.fit_on_texts(all_labels)

    # Convert labes to sequences
    label_sequences = tokenizer.texts_to_sequences(split_labels)

    # convert sequenes to a numpy array.
    ###### Using keras Tokenizer yields values that start at 1 rather than o. This will present a problem when training
    # since keras usually expects the labels to start at o. ###########

    label_sequences_np1 = np.array(label_sequences)
    label_sequences_np = label_sequences_np1 -1

    return label_sequences_np, label_sequences_np1

def create_model(num_words, embedding_dim, max_len):
    """
    creates a text clasifier model
    :param num_words: (int) size of the vocabulary for the Embedding Layer Input
    :param embedding_dim: (int) dimensionality of the Embedding layer output
    :param max_len: (int): length of the input sequences
    :return: (tf.keras Model)  the text classifier model
    """
    tensorflow.random.set_seed(123)

    model = keras.Sequential([
        keras.layers.Embedding(num_words, embedding_dim, input_length=max_len),
        keras.layers.Dropout(0.2),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(12, activation='relu'),
        keras.layers.Dense(5, activation='softmax')
    ])

    model.summary()
    model.compile(loss=keras.losses.sparse_categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

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

def visualize_3D_vectors(word_index, num_words):

    #Reverse word_index
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

    # the embedding Layer
    embedding_layer = model.layers[0]

    # Save the weights of the embedding layer
    embedding_weights = embedding_layer.get_weights()[0]

    # Print the shape. expected is (vocab_size, embedding_dim)
    print(f'Weights of the embedding layer have shape {embedding_weights.shape}')

    # Open writable files
    out_v = io.open('./em_projector/vecs.tsv', 'w', encoding='utf-8')
    out_m = io.open('./em_projector/meta.tsv', 'w', encoding='utf-8')

    for word_num in range(1, num_words):
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

    #Create sentences and labels
    sentences, labels = parse_data_from_file('./bbc-text/bbc-text.csv')

    print('ORIGINAL DATASET:\n')
    print(f'There are {len(sentences)} sentences in the dataset\n')
    print(f'The first sentence has {len(sentences[0].split())} words (after removing stopwords).\n')
    print(f'There are {len(labels)} labels in the dataset\n')
    print(f'The first 5 labels are {labels[:5]}\n\n')

    #Split Dataset
    train_sentences, train_labels, validation_sentences, validation_labels = train_val_split(sentences, labels, TRAINING_SPLIT)

    #Test the function
    print(f'There are {len(train_sentences)} sentences for training.\n')
    print(f'There are {len(train_labels)} labels for training.\n')

    print(f'There are {len(validation_sentences)} sentences for validation.\n')
    print(f'There are {len(validation_labels)} labels for validation.\n')

    #Tokenizer
    tokenizer = fit_tokenizer(train_sentences, num_words=NUM_WORDS, oov_token=OOV_TOKEN)
    word_index = tokenizer.word_index

    # Test the funtion
    print(f'Vocabulary contains {len(word_index)} words\n')
    print('<oov> token included in vocabulary' if "<OOV>" in word_index else "<OOV> token NOT included in vocabulary")

    # Sequences and padding
    train_padded_sequences= seq_and_pad(train_sentences, tokenizer, padding=PADDING, maxlen=MAXLEN)
    val_padded_sequences = seq_and_pad(validation_sentences, tokenizer, padding=PADDING, maxlen=MAXLEN)

    # Test the function
    print(f'padded training sequences have shape: {train_padded_sequences.shape}')
    print(f'padded validation sequences have shape: {val_padded_sequences.shape}')

    # Tokenize labels
    train_label_sequences_np, train_label_sequences_np1 = tokenize_labels(labels, train_labels)
    validation_label_sequences_np, validation_label_sequences_np1 = tokenize_labels(labels, validation_labels)

    print(f' First 5 labels of the training set should look like this: \n {train_label_sequences_np[:5]}\n')
    print(f' First 5 labels of the validation set should look like this: \n {validation_label_sequences_np[:5]}\n')

    print(f' Tokenized 5 labels of the training set have shape: \n {train_label_sequences_np.shape}\n')
    print(f' Tokenized 5 labels of the validation set have shape: \n {validation_label_sequences_np.shape}\n')

    ##### Buidld model #####
    model = create_model(num_words=NUM_WORDS, embedding_dim=EMBEDDING_DIM, max_len=MAXLEN)
    history = model.fit(train_padded_sequences, train_label_sequences_np,
                        validation_data=(val_padded_sequences, validation_label_sequences_np), epochs=30, verbose=1)

    plot_function(history)
    visualize_3D_vectors(word_index, num_words=NUM_WORDS)
