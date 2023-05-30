# Tokenizing the Sarcasm dataset

# Download and inspect the dataset

import wget
import os
import json

from tensorflow import keras


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

# Processing the headlines
# Initialize the tokenozer class
tokenizer = keras.preprocessing.text.Tokenizer(oov_token="<OOV>")

# Generate the word index dictionary
tokenizer.fit_on_texts(sentences)

# Print the length of the word index
word_index = tokenizer.word_index
print(f'number of words in word_index: {len(word_index)}')
print(f'the word index is: {word_index}')
print('######################################################################################')

# Generate and pad the sequences
sequences = tokenizer.texts_to_sequences(sentences)
padded_sequences = keras.preprocessing.sequence.pad_sequences(sequences, padding='post')

# Print sample padded sequences
index = 2
print(f' sample headline {sequences[index]}')
print(f' sample padded sequence {padded_sequences[index]}')

print(f' The dimension of a padded sequence is: {padded_sequences.shape}')

