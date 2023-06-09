# Training a Single Layer Neural Network with Time Series Data

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

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



def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    """
    Generate dataset windows
    :param series: (array of float) - contains the values of the time series
    :param window_size: (int) - the number of time steps to include in the feature
    :param batch_size: (int) - the batch size
    :param shuffle_buffer: (int) - buffer size to use for the shuffle method
    :return: (TF Dataset) TF Dataset containing time windows
    """
    # Generate a TF Dataset from series values
    dataset = tf.data.Dataset.from_tensor_slices(series)

    # Window the data but only take those with the specified size
    dataset = dataset.window(window_size+1, shift=1, drop_remainder=True)

    # Flatten the windows by putting its elements in a single batch
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))

    # Create tuple with features and labels
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))

    #Schuffle the windows
    dataset = dataset.shuffle(shuffle_buffer)

    # Creaste bathes of windows
    dataset = dataset.batch(batch_size=batch_size).prefetch(1)

    return  dataset

if __name__ == '__main__':
    ############### Generate Synthetic data ###############
    time = np.arange(4*365+1, dtype="float32")
    baseline = 10
    amplitude = 40
    slope = 0.05
    noise_level = 5

    # create the Series
    series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)

    # Update with noise
    series += noise(time, noise_level, seed=42)

    # Plot results
    plot_series(time, series)

    ############## Split The Dataset ###############
    split_time = 1000

    # Get the train set
    time_train = time[:split_time]
    x_train = series[:split_time]

    # Get the validation set
    time_valid = time[split_time:]
    x_valid = series[split_time:]

    plot_series(time_train, x_train)
    plot_series(time_valid, x_valid)

    ###### Prepare Features and Labels ######
    # Parameters
    window_size = 20
    batch_size = 32
    shuffle_buffer_size = 1000

    dataset = windowed_dataset(series, window_size=window_size, batch_size=batch_size, shuffle_buffer=shuffle_buffer_size)

    for windows in dataset.take(1):
        print(f'data type: {type(windows)}')
        print(f'number of elements in a tuple: {len(windows)}')
        print(f'shape of first element: {windows[0].shape}')
        print(f'shape of second element: {windows[1].shape}')

    # Build and compile the model
    layer_0 = tf.keras.layers.Dense(1, input_shape=[window_size])
    model = tf.keras.models.Sequential([layer_0])

    #print the initial layer weights
    print(f'Layer weights: {layer_0.get_weights()}')
    model.summary()
    model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(learning_rate=1e-6, momentum=0.9))
    model.fit(dataset, epochs=100)

    #Model Prediction
    # Shape of the first 20 data poits slice
    print(f'shape of series[0:20]: {series[0:10].shape}')
    #shape after addig a batch dimension
    print(f'shape of series[0:20][np.newaxis]: {series[0:10][np.newaxis].shape}')
    # shape after addig a batch dimension alternative way
    print(f'shape of series[0:20][np.newaxis]: {np.expand_dims(series[0:10], axis=0).shape}')
    # Sampe model prediction
    print(f'model prediction: {model.predict(series[0:20][np.newaxis])}')

    # prepare to ferecast data
    forecast = []

    # Use model to predict data points per window size
    for time in range(len(series) - window_size):
        forecast.append(model.predict(series[time:time + window_size][np.newaxis]))

    # Slice the points that are aligned with the validation set
    forecast = forecast[split_time-window_size:]

    # Compare number of elements in the predictions an the validationset
    print(f'length oft forecast list: {len(forecast)}')
    print(f'length oft validation set: {x_valid.shape}')

    # Visualize the results
    print(f'shape after convertion forecast tu numpy array: {np.array(forecast).shape}')
    print(f'shape after convertion forecast tu numpy array: {np.array(forecast).squeeze().shape}')

    # Convert to numpy array and drop single dimensional axes
    results = np.array(forecast).squeeze()

    plot_series(time_valid, (x_valid, results))

    ############ Computing Metrics ###########
    print(keras.metrics.mean_squared_error(x_valid, results).numpy())
    print(keras.metrics.mean_absolute_error(x_valid, results).numpy)





