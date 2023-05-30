# Training a binary classified model with the IMDB Dataset
# In this lab we will build a sentiment mode to distinguish between positive and negative movies reviews.

import tensorflow_datasets as tfds
import numpy as np
from tensorflow import  keras

import io

imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)

print(info)
print(imdb)

#get Train data
train_data, test_data = imdb['train'], imdb['test']

training_sentences = []
training_labels = []

test_sentences = []
test_labels = []

# loop over Training
for sequence, label in train_data:
    training_sentences.append(sequence.numpy().decode('utf8'))
    training_labels.append(label.numpy())

# loop over Test
for sequence, label in test_data:
    test_sentences.append(sequence.numpy().decode('utf8'))
    test_labels.append(label.numpy())

#convert labels lists into numpy array
train_labels_final = np.array(training_labels)
tests_labels_final = np.array(test_labels)

#Generate padded sequences
#Parameters
vocab_size = 10000
max_length = 120
embedding_dim = 16
trunc_type = 'post'
oov_tok = "<oov>"

def fit_tokenizer(sentences):

    """
    Instantiates the Tokenizer class
    :param sentences: (list): lower-cased sentences without stopwords
    :returns: tokeni
    """

    # Initialize the tokenozer class
    tokenizer = keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token=oov_tok)

    # Generate the word index dictionary
    tokenizer.fit_on_texts(sentences)

    # Print the length of the word index
    word_index = tokenizer.word_index

    return tokenizer, word_index

def get_padded_sequences(tokenizer, sentences):
    """
    Generates an array of token sequences and pads them to the same length
    :param tokenizer:  (Object) Tokenizer Instance containing the word-index vocabulary
    :param sentences: (list of strings) list of sentences to tokenize and pad
    :return: (array of int) tokenized sentences padded to the same length
    """

    # Convert sentences and sequences
    sequences = tokenizer.texts_to_sequences(sentences)

    #Pad the sequences using the post padding strategy
    padded_sequences = keras.utils.pad_sequences(sequences, truncating=trunc_type, maxlen=max_length)

    return padded_sequences

def tokenize_labels(labels):
    """
    Tokenizes the labels
    :param labels: labels (list of strings): labels to tekenize
    :return: label_sequences, label_word_index (list of strings, dictionary)-tokenized labels and the word index
    """
    label_tokenizer = keras.preprocessing.text.Tokenizer()
    label_tokenizer.fit_on_texts(labels)
    labels_word_index = label_tokenizer.word_index
    label_sequences = label_tokenizer.texts_to_sequences(labels)

    return label_sequences, labels_word_index

def build_model():
    model = keras.Sequential([
        keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        keras.layers.Flatten(),
        keras.layers.Dense(6, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model


def visualize_wordEmbedding(model):
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

    tokenizer_train, word_index_train = fit_tokenizer(training_sentences)
    #Generate training padded sequences
    padded_sequences_train = get_padded_sequences(tokenizer_train, training_sentences)
    #Generate testing padded sequences
    padded_sequences_test = get_padded_sequences(tokenizer_train, test_sentences)

    #numer epochs
    num_epochs = 10
    #fit model
    model = build_model()
    model.fit(padded_sequences_train, train_labels_final, epochs=num_epochs, validation_data=(padded_sequences_test, tests_labels_final))
    # Visualize Embedding
    visualize_wordEmbedding(model)

    
