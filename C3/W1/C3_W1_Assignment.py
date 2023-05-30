# Explore the BBC News archive
import csv

import wget
import os
import json

from tensorflow import keras


url = 'https://storage.googleapis.com/learning-datasets/bbc-text.csv'
if not os.listdir('./bbc-text'):
    train_files = wget.download(url, out='./bbc-text')

# Graded Function: removing stop word
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

def fit_tokenizer(sentences):

    """
    Instantiates the Tokenizer class
    :param sentences: (list): lower-cased sentences without stopwords
    :returns: tokeni
    """

    # Initialize the tokenozer class
    tokenizer = keras.preprocessing.text.Tokenizer(oov_token="<OOV>")

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
    padded_sequences = keras.preprocessing.sequence.pad_sequences(sequences, padding='post')

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

if __name__ == '__main__':
    sentence_without_sw = remove_stopwords("I am about to go to the store and get any snack")
    print(sentence_without_sw)

    #Create sentences and labels
    sentences, labels = parse_data_from_file('./bbc-text/bbc-text.csv')

    print('ORIGINAL DATASET:\n')
    print(f'There are {len(sentences)} sentences in the dataset\n')
    print(f'The first sentence has {len(sentences[0].split())} words (after removing stopwords).\n')
    print(f'There are {len(labels)} labels in the dataset\n')
    print(f'The first 5 labels are {labels[:5]}\n\n')

    # Create the Tokenizer object and fit it on sentences
    tokenizer, word_index = fit_tokenizer(sentences)
    print(f'Vocabulary contains {len(word_index)} words\n')
    print('<OOV> token included in vocabulary' if "<OOV>" in word_index else '<OOV> token NOT included in vocabulary')

    # Create the padded_sequences
    padded_sequences = get_padded_sequences(tokenizer, sentences)
    print(f"First added sequence looks like this: \n\n{padded_sequences[0]}")
    print(f"Nummpy array of  all sequences has shape: \n\n{padded_sequences.shape}")
    print(f"This  means they are {padded_sequences.shape[0]} sequences in total and each one has a size of {padded_sequences.shape[1]}")

    # Labels tokenizing
    labels_sequences, labels_word_index = tokenize_labels(labels)
    print(f'Vocabulary of labels look like this {labels_word_index}')
    print(f'Fist ten sequences {labels_sequences[:10]}')




