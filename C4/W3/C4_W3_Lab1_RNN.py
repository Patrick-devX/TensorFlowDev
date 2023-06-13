#Using a Simple RNN for forecastin

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass


def plot_series(time, series, format="-", start=0, end=None):
    """
    Visualize time series data
    :param time: (array of int) - contains the time steps
    :param series: (array of int) - contains measurements for each time step
    :param format: (string) . line style when plotting the graph
    :param start: (int) - first time step to plot
    :param end:  (int) - last time step to plot
    :param label: (list of strings) - tag for the line
    :return:
    """
    # Set up dimensions of the graph figure
    plt.figure(figsize=(10, 6))

    if type(series) is tuple:
        for series_num in series:
            #Plot the time series data
            plt.plot(time[start:end], series_num[start:end], format)
    else:
        # Plot the tieme series data
        plt.plot(time[start:end], series[start:end], format)

    # label the x-axis
    plt.xlabel('Time')
    # Label y-axis
    plt.ylabel('Value')
    plt.grid('True')
    plt.show()

def trend(time, slope=0):
    """
    Generate synthetic data that follows a straight line given a slope value
    :param time: (array of int) - contains the time steps
    :param slope: (float) - determines the direction and steepness of the line
    :return: series (array of float) - measurements that follow a straight line
    """
    # Compute the linear series given the slope
    series = slope * time

    return series

def seasonal_pattern(season_time):
    """
    Just an arbitrary pattern, sou can change it if you wisch
    :param season_time: (array of float) - contains the measurements per time step
    :return: data_pattern (array of float) - contains revised measurment values according to the defined pattern
    """
    #Generate the values using an arbitrary pattern
    data_pattern = np.where(season_time <0.4, np.cos(season_time*2*np.pi), 1/np.exp(3*season_time))

    return data_pattern

def seasonality(time, period, amplitude=1, phase=0):
    """
    repeat the same pattern at each period
    :param time: (array of inf) contains the time steps
    :param period: (int) number of time steps before the pattern repeat
    :param amplitude: (int) peak measured value in a period
    :param phase: number of time steps to shift the mesured values
    :return: data_pattern (array of float) -seasonal data scaled by the defined amplitude
    """
    # Define the measured values per perid
    season_time = ((time+phase) % period) / period

    #Generate the seasonal data scaled by defined amplitude
    data_pattern = amplitude * seasonal_pattern(season_time)

    return data_pattern

def noise(time, noise_level=1, seed=None):
    """
    Generates a normally distributed noisy signal
    :param time:  (array of int) - contains the time steps
    :param noise_level: (float) - scaling factor for the generated signal
    :param seed: (int) number generator seed for repeatability
    :return: noise (array of float) - the noisy signal
    """

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

@dataclass
class G:
    TIME, SERIES = generate_time_series()
    SPLITE_TIME = 1000
    WINDOW_SZE = 20
    BATCH_SIZE = 32
    SHUFFLE_BUFFER_SIZE = 1000

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

def build_model():
    model_tune = tf.keras.models.Sequential([
        tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[G.WINDOW_SZE]),
        tf.keras.layers.SimpleRNN(40, return_sequences=True),
        tf.keras.layers.SimpleRNN(40),
        tf.keras.layers.Dense(1),
        tf.keras.layers.Lambda(lambda x: x * 100)
    ])

    lr_shedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10**(epoch/20))



    model_tune.summary()
    model_tune.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.SGD(momentum=0.9), metrics=['mae'])

    return model_tune, lr_shedule

def build_model_tuned():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[G.WINDOW_SZE]),
        tf.keras.layers.SimpleRNN(40, return_sequences=True),
        tf.keras.layers.SimpleRNN(40),
        tf.keras.layers.Dense(1),
        tf.keras.layers.Lambda(lambda x: x * 100)
    ])
    model.summary()
    model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.SGD(learning_rate=1e-6, momentum=0.9))

    return model

def plot_results_training(train_history):
    # Define the learning rate array
    lrs = 1e-8*(10**(np.arange(100)/20))

    #Set fig size
    plt.figure(figsize=(10, 6))
    plt.grid(True)
    # Plot the loss in log scale
    plt.semilogx(lrs, train_history.history['loss'])
    # Increase the tickmarks size
    plt.tick_params('both', length=10, width=1, which='both')
    # Set the plot boundaries
    plt.axis([1e-8, 1e-3, 0, 50])
    plt.show()

def forecasting(model, series, window_size, time_valid, x_valid, split_time=1000):
    # Initialize list
    forecast = []

    # Reduce the original series[0,1,2, ...1460] --> [1000, 1001, 1002, ...1460]: 461 Elements
    forecast_series = series[split_time - window_size:]

    #Use the model to predict data points per window size
    for time in range(len(forecast_series) - window_size):
        forecast.append(model.predict(forecast_series[time:time + window_size][np.newaxis]))

    # Convert to numpy array and drop single dimensional axes
    results = np.array(forecast).squeeze()

    # Plot the results
    plot_series(time_valid, (x_valid, results))

    return results

def model_forecast(model, series, window_size, batch_size):
    """
    Use a trained model to generate predictions
    :param model: (TF Keras model) - model that accept windows
    :param series: (array of float) - contains the values of the times series
    :param window_size: (int) - the number of time steps to include in the window
    :param batch_size: (int) - the batch size
    :return: (numpy array) - array containing predictions
    """
    # Generate a TF Dataset from the series values
    dataset = tf.data.Dataset.from_tensor_slices(series)

    #Window the data but only take those with the specified size
    dataset = dataset.window(window_size, shift=1, drop_remainder=True)

    # Flatten the windows by putting its elements in a single batch
    dataset = dataset.flat_map(lambda w: w.batch(window_size))

    # Crate baches of windows
    dataset = dataset.batch(batch_size).prefetch(1)

    # Get Predictions
    forecast = model.predict(dataset)

    return forecast


if __name__ == '__main__':
    dataset = windowed_dataset(G.SERIES, window_size=G.WINDOW_SZE, batch_size=G.BATCH_SIZE,
                                    shuffle_buffer=G.SHUFFLE_BUFFER_SIZE)

    # Print shape of features and labels
    for window in dataset.take(1):
        print(f'shape of feature: {window[0].shape}')
        print(f'shape of label: {window[1].shape}')

    model_tune, lr_shedule = build_model()
    hiytory = model_tune.fit(dataset, epochs=100, callbacks=[lr_shedule])

    # Pick zhe best learning rate by plotting the loss over the learning rate
    plot_results_training(hiytory)

    # The loss start getting instable at lr =  2e-8, so will train the model again with this lr
    # We will also minimize the sensitity to outliers by using the Huber Loss
    model = build_model_tuned()
    hiytory = model_tune.fit(dataset, epochs=100)


    time_train, x_train, time_valid, x_valid = train_val_split(G.TIME, G.SERIES, split_time=G.SPLITE_TIME)

    results = forecasting(model, G.SERIES, G.WINDOW_SZE, time_valid, x_valid, split_time=1000)

    # Fast forecastion
    # get the approprate size
    forecast_series = G.SERIES[G.SPLITE_TIME-G.WINDOW_SZE: -1]

    # Generate predictions
    forecast = model_forecast(model, forecast_series, G.WINDOW_SZE, G.BATCH_SIZE)

    # Drop single dimensional axis
    forecast = forecast.squeeze()

    plot_series(time_valid, (x_valid, forecast))
