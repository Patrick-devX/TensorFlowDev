# Predicting Time Series

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from dataclasses import dataclass

def plot_series(time, series, format="-", start=0, end=None):
    plt.figure(figsize=(10, 6))

    if type(series) is tuple:
        for series_num in series:
            # Plot the time series data
            plt.plot(time[start:end], series_num[start:end], format)
    else:
        # Plot the tieme series data
        plt.plot(time[start:end], series[start:end], format)

    plt.grid(True)
    plt.show()

def trend(time, slope=0):
    # Compute the linear series given the slope
    series = slope * time

    return series

def seasonal_pattern(season_time):
    #Generate the values using an arbitrary pattern
    data_pattern = np.where(season_time <0.1, np.cos(season_time*6*np.pi), 2/np.exp(9*season_time))

    return data_pattern

def seasonality(time, period, amplitude=1, phase=0):
    # Define the measured values per perid
    season_time = ((time+phase) % period) / period

    #Generate the seasonal data scaled by defined amplitude
    data_pattern = amplitude * seasonal_pattern(season_time)

    return data_pattern

def noise(time, noise_level=1, seed=None):
    # Initialize the random number generator
    rnd = np.random.RandomState(seed)

    #Generate a random number for each time
    noise = rnd.randn(len(time))*noise_level

    return noise

def generate_time_series():
    # The time dimension or the x-coordinate of time series
    time = np.arange(4 * 365 + 1, dtype="float32")

    slope = 0.05
    baseline = 10
    amplitude = 40
    # create the Series
    series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)

    noise_level = 5
    # Update with noise
    series += noise(time, noise_level, seed=42)

    return time, series

# Save all Global variables within the G class (G stands forglobal)
@dataclass
class G:
    TIME, SERIES = generate_time_series()
    SPLITE_TIME = 1000
    WINDOW_SZE = 1
    BATCH_SIZE = 5
    SHUFFLE_BUFFER_SIZE = 1

def train_val_split(time, series, split_time=G.SPLITE_TIME):
    # Get the train set
    time_train = time[:split_time]
    x_train = series[:split_time]

    # Get the validation set
    time_valid = time[split_time:]
    x_valid = series[split_time:]

    plot_series(time_train, x_train)
    plot_series(time_valid, x_valid)

    return time_train, x_train, time_valid, x_valid

def windowed_dataset(series, window_size=G.WINDOW_SZE, batch_size=G.BATCH_SIZE, shuffle_buffer=G.SHUFFLE_BUFFER_SIZE):
    # create dataset from the series
    dataset = tf.data.Dataset.from_tensor_slices(series)

    # Slice the dataset into the appropriate windows
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)

    # Flatten the windows by putting its elements in a single batch
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))

    #shuffle the data
    dataset = dataset.shuffle(shuffle_buffer)

    # Split into features and labels
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))

    # Create bathes of windows
    dataset = dataset.batch(batch_size=batch_size).prefetch(1)

    return dataset

if __name__ == '__main__':
    # Plot the genaratd series
    #plt.figure(figsize=(10, 6))
    plot_series(G.TIME, G.SERIES)
    plt.grid(True)
    plt.show()

    test_dataset = windowed_dataset(G.SERIES, window_size=G.WINDOW_SZE, batch_size=G.BATCH_SIZE, shuffle_buffer=G.SHUFFLE_BUFFER_SIZE)

    # Get the first batch of the dataset
    batch_of_features, batch_of_labels = next((iter(test_dataset)))

    time_train, x_train, time_valid, x_valid = train_val_split(G.TIME, G.SERIES, split_time=G.SPLITE_TIME)

    print(f'batch of features has type: {type(batch_of_features)}\n')
    print(f'batch of labels has type: {type(batch_of_labels)}\n')
    print(f'batch of features has shape: {batch_of_features.shape}\n')
    print(f'batch of labels has shape: {batch_of_labels.shape}\n')
    print(f'batch of features is equal to first five elements in the series: {np.allclose(batch_of_features.numpy().flatten(), x_train[:5])}\n')
    print(f'batch of label equal to first five labels: {np.allclose(batch_of_features.numpy().flatten(), x_train[1:6])}')