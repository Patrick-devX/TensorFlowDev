from tensorflow import keras

#Define sentences
sentences = ['i love my dog', 'i love my cat', 'you love my dog', 'Do you think my dog is amazing? ']

#Initialize zhe Tokernizer class
tockenizer = keras.preprocessing.text.Tokenizer(num_words=100, oov_token='<OOV>') # num_words can be set to 1 it does n matter.

#Generate indices for each word in the corpus
tockenizer.fit_on_texts(sentences)

#Get the indices and print ist
word_index = tockenizer.word_index
print(word_index)

# Generate list of token sequences
sequences = tockenizer.texts_to_sequences(sentences)
print(f'\nSequences = {sequences}')

# Padding
# Pad the sequence to a uniform length
padded_sequence = keras.preprocessing.sequence.pad_sequences(sequences)
print(f'\nPadded Sequences default = {padded_sequence}')

padded_sequence_ml = keras.preprocessing.sequence.pad_sequences(sequences, maxlen=5)
print(f'\nPadded Sequences mit maxlen = 5 : {padded_sequence_ml}')