# Tokernizer Basic

from tensorflow import keras

#Define sentences
sentences = ['i love my dog', 'i love my cat']

#Initialize zhe Tokernizer class
tockenizer = keras.preprocessing.text.Tokenizer(num_words=100) # num_words can be set to 1 it does n matter.

#Generate indices for each word in the corpus
tockenizer.fit_on_texts(sentences)

#Get the indices and print ist
word_index = tockenizer.word_index
print(word_index)