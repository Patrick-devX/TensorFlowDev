# Generating Text with Neural Networks

from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data="In the town of Athy one Jeremy Lanigan \n Battered away til he hadnt a pound. \nHis father died and made him a man again \n Left him a farm and ten acres of ground. \nHe gave a grand party for friends and relations \nWho didnt forget him when come to the wall, \nAnd if youll but listen Ill make your eyes glisten \nOf the rows and the ructions of Lanigans Ball. \nMyself to be sure got free invitation, \nFor all the nice girls and boys I might ask, \nAnd just in a minute both friends and relations \nWere dancing round merry as bees round a cask. \nJudy ODaly, that nice little milliner, \nShe tipped me a wink for to give her a call, \nAnd I soon arrived with Peggy McGilligan \nJust in time for Lanigans Ball. \nThere were lashings of punch and wine for the ladies, \nPotatoes and cakes; there was bacon and tea, \nThere were the Nolans, Dolans, OGradys \nCourting the girls and dancing away. \nSongs they went round as plenty as water, \nThe harp that once sounded in Taras old hall,\nSweet Nelly Gray and The Rat Catchers Daughter,\nAll singing together at Lanigans Ball. \nThey were doing all kinds of nonsensical polkas \nAll round the room in a whirligig. \nJulia and I, we banished their nonsense \nAnd tipped them the twist of a reel and a jig. \nAch mavrone, how the girls got all mad at me \nDanced til youd think the ceiling would fall. \nFor I spent three weeks at Brooks Academy \nLearning new steps for Lanigans Ball. \nThree long weeks I spent up in Dublin, \nThree long weeks to learn nothing at all,\n Three long weeks I spent up in Dublin, \nLearning new steps for Lanigans Ball. \nShe stepped out and I stepped in again, \nI stepped out and she stepped in again, \nShe stepped out and I stepped in again, \nLearning new steps for Lanigans Ball. \nBoys were all merry and the girls they were hearty \nAnd danced all around in couples and groups, \nTil an accident happened, young Terrance McCarthy \nPut his right leg through miss Finnertys hoops. \nPoor creature fainted and cried Meelia murther, \nCalled for her brothers and gathered them all. \nCarmody swore that hed go no further \nTil he had satisfaction at Lanigans Ball. \nIn the midst of the row miss Kerrigan fainted, \nHer cheeks at the same time as red as a rose. \nSome of the lads declared she was painted, \nShe took a small drop too much, I suppose. \nHer sweetheart, Ned Morgan, so powerful and able, \nWhen he saw his fair colleen stretched out by the wall, \nTore the left leg from under the table \nAnd smashed all the Chaneys at Lanigans Ball. \nBoys, oh boys, twas then there were runctions. \nMyself got a lick from big Phelim McHugh. \nI soon replied to his introduction \nAnd kicked up a terrible hullabaloo. \nOld Casey, the piper, was near being strangled. \nThey squeezed up his pipes, bellows, chanters and all. \nThe girls, in their ribbons, they got all entangled \nAnd that put an end to Lanigans Ball."

# Split the long string per line and put in a list
corpus = data.lower().split("\n")

# Preview the result
print(corpus)

# Initialize the Tokenizer Class
tokenizer = keras.preprocessing.text.Tokenizer()

#Generate the word index dictionary
tokenizer.fit_on_texts(corpus)

# Define the total words. You add 1 for the index '0' which is just tge padding tocken
total_words = len(tokenizer.word_index) +1

print(f'word index dictionary: {tokenizer.word_index}')
print(f'total words: {total_words}')

# Processing the Dataset

# Initialize the sequences list
input_sequence = []
for line in corpus:
    #tokenize the current line
    token_list = tokenizer.texts_to_sequences([line])[0]
    # Loop over the line several times to generate the subphrases
    for i in range(1, len(token_list)):
        # Generate the subphrase
        n_gram_sequence = token_list[:i+1]
        #Append the subphrase to the sequence list
        input_sequence.append(n_gram_sequence)

# Get the length of the longest line in corpus
max_sequence_len = max([len(x) for x in input_sequence])

# Pad all sequences
padded_input_sequences = np.array(keras.preprocessing.sequence.pad_sequences(input_sequence, maxlen=max_sequence_len, padding='pre'))

# Creating input and labels by splitting the last token in the supprases
inputs, labels = padded_input_sequences[:, :-1], padded_input_sequences[:, -1]

# convert the labels into one-hot arrays
labels_ys = keras.utils.to_categorical(labels, num_classes=total_words)

######################################################################################
 # Lets see the result for the first line of the song
 # Get sample sentence
sentence = corpus[0].split()

 # Initialize token list
token_list = []

 # look up the indices of each and append to the list
for word in sentence:
    print(word)
    token_list.append(tokenizer.word_index[word])
print(token_list)

######################################################################################
# Pick element
elemen_number = 6

# Print token list and phrase
print(f'token list: {inputs[elemen_number]}')
print(f'decoded to text: {tokenizer.sequences_to_texts([inputs[elemen_number]])}')

# Print label
print(f'one-hot label: {labels_ys[elemen_number]}')
print(f'index of label: {np.argmax(labels_ys[elemen_number])}')

# Print element
elemen_number = 5

# Print token list an phrase
print(f'token list: {inputs[elemen_number]}')
print(f'decoded to text: {tokenizer.sequences_to_texts([inputs[elemen_number]])}')

# Print label
print(f'one-hot label: {labels_ys[elemen_number]}')
print(f'index of label: {np.argmax(labels_ys[elemen_number])}')

#################################################################################
# Build Model

model = keras.Sequential([
    keras.layers.Embedding(total_words, 64, input_length=max_sequence_len- 1),
    keras.layers.Bidirectional(keras.layers.LSTM(20)),
    keras.layers.Dense(total_words, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
history = model.fit(inputs, labels_ys, epochs=500)

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

plot_function(history)

################################################################################
# Generating Text

# Define seed text
seed_text = "Laurence went to Dublin"

# Define total words to predict
next_words = 100

# Loop until desired length is reached
for _ in range(next_words):

    # convert the seed text to a token sequence
    token_list = tokenizer.texts_to_sequences([seed_text])[0]

    # Padd the sequence
    token_list = keras.preprocessing.sequence.pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')

    # Feed to the model an get the probabilities of each index
    probabilities = model.predict(token_list)

    predicted = np.argmax(probabilities, axis=-1)[0]

    # Ignore if index is 0 because tha is just padding
    if predicted != 0:
        output_word = tokenizer.index_word[predicted]
    # Combine with the seed text
    seed_text += " " + output_word
print(seed_text)


################################################################################
# Generating Text

# Define seed text
seed_text = "Laurence went to Dublin"

# Define total words to predict
next_words = 100

# Loop until desired length is reached
for _ in range(next_words):

    # convert the seed text to a token sequence
    token_list = tokenizer.texts_to_sequences([seed_text])[0]

    # Padd the sequence
    token_list = keras.preprocessing.sequence.pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')

    # Feed to the model an get the probabilities of each index
    probabilities = model.predict(token_list)

    # Pick a random number from [1,2,3]
    choice = np.random.choice([1, 2, 3])

    predicted = np.argsort(probabilities)[0][-choice]

    # Ignore if index is 0 because tha is just padding
    if predicted != 0:
        output_word = tokenizer.index_word[predicted]
    # Combine with the seed text
    seed_text += " " + output_word
print(seed_text)