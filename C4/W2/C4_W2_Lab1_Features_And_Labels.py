# Preparing Time Series Features and Labels

from tensorflow import keras
import tensorflow as tf


# Create a simple Dataset
# Generate a tf dataset with 10 elements (i.e number 0 to 9)
dataset = tf.data.Dataset.range(10)

#Preview the results
#for value in dataset:
    #print(value.numpy())

## Windowing the data
dataset = tf.data.Dataset.range(10)

#Window data
dataset = dataset.window(size=5, shift=1, drop_remainder=True)

#print the result
for window_data in dataset:
    #print(window_data)
    print([item.numpy() for item in window_data])

## Flatten the Window
dataset = tf.data.Dataset.range(10)

#Window data
dataset = dataset.window(size=5, shift=1, drop_remainder=True)

# Flatten the windows by puting its elements in a single batch
dataset = dataset.flat_map(lambda window: window.batch(5))

print(" #########  Flatten the Window #########")
#print the result
for window in dataset:
    print(window.numpy())

### Group into features and labels
# Generate a tf dataset with 10 elements (i.e number 0 to 9)
dataset = tf.data.Dataset.range(10)

#Window data
dataset = dataset.window(size=5, shift=1, drop_remainder=True)

# Flatten the windows by puting its elements in a single batch
dataset = dataset.flat_map(lambda window: window.batch(5))

# Create tuples with features (first four elements of window) and labels (last element)
dataset = dataset.map(lambda window: (window[:-1], window[-1]))

# Shuffle the windows
dataset = dataset.shuffle(buffer_size=10)

# Create batches of windoes
dataset = dataset.batch(2).prefetch(1)

print(" #########  Group into features and labels #########")
# Print the resutlts
for x, y in dataset:
    print(f'x = {x.numpy()}')
    print(f'y = {y.numpy()}')
    print()


