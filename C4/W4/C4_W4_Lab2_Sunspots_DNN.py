# Predicting Sunsplots with Neural Networks (DNN only)

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv
import wget
import os

def plot_series(x, y, format='-', start=0, end=None, title=None,
                xlabel=None, ylabel=None, legend=None):
    """
    Visualizes time series data
    :param x: (array of int) - contains values for the x-axis
    :param y: (array of int or tuple of array) - contains values for the x-axis
    :param format: (string) - line style when plottig the graph
    :param start: (int) - first time step to plot
    :param end: (int) - Last time step to plot
    :param title: (string) - title of plot
    :param xlabel: (string) - label for the x axis
    :param ylabel: (string) - label for the y -axis
     :param legend: (kist of strings) -legend for the plot
    :return:
    """
    # set up dimensions of figure
    plt.figure(figsize=(10, 6))

    # Check if there are more than 2 series to plot
    if type(y) is tuple:
        for y_current in y:
            plt.plot(x[start:end], y_current[start:end], format)
    else:
        plt.plot(x[start:end], y[start:end], format)
        # Label the x.axis
        plt.xlabel(xlabel)
        # Label the y axis
        plt.ylabel(ylabel)
        # Set the legend
    if legend:
        plt.legend(legend)
    #set title
    plt.title(title)
    plt.grid(True)
    plt.show()

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    """
    Generates dataset windows
    :param series: (array of float) - contains the values of the time series
    :param window_size: (int) - the number of time steps to include in the feature
    :param batch_size: (int) - the batch size
    :param shuffle_buffer: (int) - buffer size to use for the schuffle method
    :return: (TF Dataset) - TF Dataset containing time windows
    """
    # Generate a TF Dataset from the series values
    dataset = tf.data.Dataset.from_tensor_slices(series)

    # Window the data but only take those with the specified size
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)

    # Flatten the windows by putting its elements in a single batch
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))

    # Creates tuples with features and labels
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))

    # shuffle the windows
    dataset = dataset.shuffle(shuffle_buffer)

    # create batches of windows
    dataset = dataset.batch(batch_size).prefetch(1)

    return dataset

def build_model(window_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(30, input_shape=[window_size], activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.summary()

    return model

def plot_loss_learning_rate(train_history):
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

def forecasting(model, series, window_size, time_valid, x_valid,  split_time=1000):
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

if __name__ == '__main__':

    url_train = 'https://storage.googleapis.com/tensorflow-1-public/course4/Sunspots.csv'

    if not os.listdir('./Sunspots') or not os.listdir('./Sunspots'):
        train_files = wget.download(url_train, out='./Sunspots')

    time_step = []
    sunspots = []
    # Open the csv file
    with open('./sunspots/Sunspots.csv') as csv_file:
        # Initialize reader
        reader = csv.reader(csv_file, delimiter=',')
        #Skip the first line
        next(reader)
        #Append row and sunspot number to list
        for row in reader:
            time_step.append(int(row[0]))
            sunspots.append(float(row[2]))
    # Convert list to numpy arrays
    time = np.array(time_step)
    series = np.array(sunspots)
    # Preview the data
    plot_series(time, series, xlabel='Month', ylabel='Montly Mean Total Sunspot Number')

    # Split the Dataset
    split_time = 3000
    #Train set
    time_train = time[:split_time]
    x_train = series[:split_time]
    #Validation set
    time_val = time[split_time:]
    x_val = series[split_time:]

    # Prepare Features and Labels
    window_size = 30
    batch_size = 32
    shuffle_buffer_size = 1000
    #generate dataset window
    dataset = windowed_dataset(series=series, window_size=window_size, batch_size=batch_size, shuffle_buffer=shuffle_buffer_size)

    #Tune the learning rate
    model = build_model(window_size)
    #set the learning rate scheduler
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10**(epoch/20))
    # Initialize the optimizer
    optimizer =tf.keras.optimizers.SGD(momentum=0.9)
    #set the training parameters
    model.compile(loss=tf.keras.losses.Huber(), optimizer=optimizer)
    # train model
    history = model.fit(dataset, epochs=100, callbacks=[lr_scheduler])
    # Plot loss vs learning_rate
    plot_loss_learning_rate(history)

    #Reset states generated by keras
    model = build_model(window_size=window_size)
    # Set the learning rate
    learning_rate = 2e-5
    #set the optimizer
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    model.compile(loss=tf.keras.losses.Huber(), optimizer=optimizer, metrics=['mae'])
    history = model.fit(dataset, epochs=100)

    # Forecasting
    results = forecasting(model, series, window_size, time_val, x_val, split_time=split_time)

    #Compute MAE
    print(tf.keras.metrics.mean_absolute_error(x_val, results).numpy())



