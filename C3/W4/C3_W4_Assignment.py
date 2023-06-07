# Predicting the next word: Shakespeare Dataset

import gdown
import os
import matplotlib.pyplot as plt
import numpy as np

from tensorflow import keras

url_train = 'https://drive.google.com/uc?id=108jAePKK4R3BVYBbYJZ32JWUwxeMg20K'

if not os.listdir('./shakespeare'):
    train_files = gdown.download(url_train, output='./shakespeare/sonnet.txt')

# Read data
with open ('./shakespeare/sonnet.txt') as f:
    data = f.read()

# Convert to lower case and save as list
corpus = data.lower().split("\n")

print(f'There are {len(corpus)} line of sonnets\n')
print(f'The first 5 lines look like this:\n')
for i in range(5):
    print(corpus[i])


def tokenizing_text(corpus):
    """
    Fittig a tokenizer to the corpus
    :param corpus: (list of string)
    :return: tokenizer
    """
    tokenizer = keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(corpus)

    return tokenizer

def generating_ngrams(corpus, tokenizer):
    """
    Generates a list of n-gram sequences
    :param corpus: (list of string) lines of text to genarate n-gram for
    :param tokenizer: (object) an instance of the Tokenizer class containing the word index dictionary
    :return: input_sequences (list of int) the n-gram sequences for each line in the corpus
    """
    input_sequence = []

    for line in corpus:
        # tokenize the current line
        token_list = tokenizer.texts_to_sequences([line])[0]
        # Loop over the line several times to generate the subphrases
        for i in range(1, len(token_list)):
            # Generate the subphrase
            n_gram_sequence = token_list[:i + 1]
            # Append the subphrase to the sequence list
            input_sequence.append(n_gram_sequence)

    return input_sequence

def pad_sequences(input_sequences, maxlen):
    """
    Pad tokenized sequences to the same length
    :param input_sequences: (list of int) tokenized sequences to pad
    :param maxlen: maximum length of the token sequences
    :return: padded_sequences (array of int): tokenized sequences padded to the same length
    """
    padded_sequences = keras.preprocessing.sequence.pad_sequences(input_sequences, padding='pre', maxlen=maxlen)

    return padded_sequences

def features_and_labels(padded_input_sequences, total_words):
    """
    Generate features and labels from n-grams
    :param input_sequences: (list of int) sequences to split features and labels from
    :param total_words: (int) vocabulary_size
    :return: features, one-hot-labels (array of int, array of int) arrays of features and one-hot encoded labels
    """
    features = padded_input_sequences[:, :-1]
    labels = padded_input_sequences[:, -1]
    labels = keras.utils.to_categorical(labels, num_classes=total_words)

    return features, labels

def create_model(total_words, max_sequence_length):

    model = keras.Sequential([
        keras.layers.Embedding(total_words, output_dim=100, input_length=max_sequence_length-1),
        keras.layers.Bidirectional(keras.layers.LSTM(200)),
        keras.layers.Dense(total_words, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model

def text_generator(model, tokenizer):
    #seed text
    seed_text = "Help me Obi Wan Kenobi, you're my only hope"
    next_words= 100

    for _ in range(next_words):
        # Convert the seed text into sequences
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        #pad the sequences
        padded_token_list = keras.preprocessing.sequence.pad_sequences([token_list], maxlen=len(tokenizer.word_index)-1, padding='pre')
        # Feed to the model an get the probabilities of each index
        probabilities = model.predict(padded_token_list)

        # Pick a random number from [1,2,3]
        choice = np.random.choice([1, 2, 3])

        predicted = np.argsort(probabilities)[0][-choice]

        # Ignore if index is 0 because tha is just padding
        if predicted != 0:
            output_word = tokenizer.index_word[predicted]
        # Combine with the seed text
        seed_text += " " + output_word
    return seed_text


def plot_function(history):
    acc = history.history['accuracy']
    #val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    #val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r', label='Training accuracy')
    #plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.grid(True)
    plt.legend()
    plt.figure()

    plt.plot(epochs, loss, 'r', label='Training loss')
    #plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == '__main__':

    tokenizer = tokenizing_text(corpus)

    corpus[0]
    tokenizer.texts_to_sequences([corpus[0]])[0]

    # Get input sequencs in n-grams
    input_sequences = generating_ngrams(corpus, tokenizer)

    # Test the funtion
    first_example_sequence = generating_ngrams([corpus[0]], tokenizer=tokenizer)
    print(f'n_grams sequences for first example look like this: \n')
    print(first_example_sequence)


    first_3_example_sequence = generating_ngrams(corpus[1:4], tokenizer=tokenizer)
    print(f'n_grams sequences for first example look like this: \n')
    print(first_3_example_sequence)


    # Save maxsequence length
    max_sequence_len = max([len(x) for x in input_sequences])

    #padded sequences
    first_padded_sequences = pad_sequences(first_example_sequence, max([len(x) for x in first_example_sequence]))
    print(first_padded_sequences)

    padded_sequences = pad_sequences(input_sequences, max([len(x) for x in input_sequences]))
    print(f'padded corpus shape: {padded_sequences.shape}')

    # features and labels
    features, labels = features_and_labels(padded_sequences, len(tokenizer.word_index)+1)

    print(f'features have shape: {features.shape}')
    print(f'labels have shape: {labels.shape}')

    #Model
    model = create_model(len(tokenizer.word_index)+1, max([len(x) for x in padded_sequences]))
    history = model.fit(features, labels, epochs=50, verbose=1)
    plot_function(history)





