# Subword Tokenization with the IMDB Reviews Dataset
import tensorflow_datasets as tfds
from tensorflow import keras
import matplotlib.pyplot as plt

# Plain text default config
imdb_plaintext, info_plaintext = tfds.load("imdb_reviews", with_info=True, as_supervised=True)

#Subword encoded pretokenized dataset
imdb_subwords, info_subwords = tfds.load("imdb_reviews/subwords8k", with_info=True, as_supervised=True)

print(info_plaintext.features)

# Take 2 Training examples and print the text feature
for example in imdb_plaintext['train'].take(2):
    print(example[0].numpy())

print(info_subwords.features)

# Take 2 Training examples and print the text feature
for example in imdb_subwords['train'].take(2):
    print(example[0].numpy())

# Get the subwords encoder
tokenizer_subwords = info_subwords.features['text'].encoder

# Take 2 Training examples and print the text feature
for example in imdb_subwords['train'].take(2):
    print(tokenizer_subwords.decode(example[0]))

# Subword Text Encoding

#Get The train set
train_data = imdb_plaintext['train']

training_sentences = []
for s, _ in train_data:
    training_sentences.append(s.numpy().decode('utf8'))

# Parameters
vocab_size = 10000
oov_tok = "<OOV>"
BUFFER_SIZE = 10000
BATCH_SIZE = 64

def data_processing(sentences, oov_tok, vocab_size):

    # Initialize tockenizer
    tokenizer = keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token=oov_tok)

    # Generate the word index dictionary
    tokenizer.fit_on_texts(sentences)
    word_index = tokenizer.word_index

    # Generate and pad the sequences
    sequences = tokenizer.texts_to_sequences(sentences)

    return sequences,  tokenizer

def build_model():

    model = keras.Sequential([
        keras.layers.Embedding(vocab_size, embedding_dim),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(6, activation='relu'),
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

if __name__ == '__main__':
    sequences, tokenizer_plaintext = data_processing(sentences=training_sentences, oov_tok=oov_tok, vocab_size=vocab_size)
    decoded_part_sequence = tokenizer_plaintext.sequences_to_texts(sequences[0:1])
    print(decoded_part_sequence)
    print(len(tokenizer_plaintext.word_index))
    print(tokenizer_subwords.subwords)
    tokenizer_string = tokenizer_subwords.encode(training_sentences[0])
    print(tokenizer_string)

    #decode the sequence
    original_string = tokenizer_subwords.decode(tokenizer_string)
    #print(original_string)

    ####### Example #######
    sample_string = "TensorFlow, from basic to mastery"

    #Encode using plaintext tokenizer
    tokenized_string = tokenizer_plaintext.texts_to_sequences([sample_string])
    print(f"The tokenized String is : {tokenized_string}")

    #Decode it back using plaintext tokenizer
    original_string2 = tokenizer_plaintext.sequences_to_texts(tokenized_string)
    print(f"The original string is: {original_string2}")

    #***** Comparison to the subword text encoder *****
    tokenized_string_subword = tokenizer_subwords.encode(sample_string)
    print(f'With the subword tokenized string is: {tokenized_string_subword}')

    # Decode and print the results
    original_string_subword = tokenizer_subwords.decode(tokenized_string_subword)
    print(f'the original string with subword is: {original_string_subword}')

    # GEt TRain and Test data
    train_data, test_data = imdb_subwords['train'], imdb_subwords['test']

    # Schuffle the training data
    train_dataset = train_data.shuffle(BUFFER_SIZE)

    # Batch and pad the dataset to the maximum length of the sequence
    train_dataset = train_dataset.padded_batch(BATCH_SIZE)
    test_dataset = test_data.padded_batch(BATCH_SIZE)

    #get model
    num_epochs = 10
    embedding_dim = 64
    vocab_size = tokenizer_subwords.vocab_size

    model = build_model()

    # Starting training
    history = model.fit(train_dataset, epochs=num_epochs, validation_data=test_dataset)

    plot_function(history)


