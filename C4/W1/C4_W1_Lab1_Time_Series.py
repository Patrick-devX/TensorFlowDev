# Introduction to Time Sieries
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

def plot_series(time, series, format="-", start=0, end=None, label=None):
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

    #Plot the time sieries data
    plt.plot(time[start:end], series[start:end], format)

    # Label x-axis
    plt.xlabel('Time')
    plt.ylabel('Value')

    if label:
        plt.legend(fontsize=14, labels = label)
    plt.grid(True)
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

def moving_average_forecast(series, window_size):
    """
    Generates a moving average forecast
    :param series: (array of float) - contains the values of the tome series
    :param window_size: (int) - the number of time steps to compute the average for
    :return: forecast (array of float) - the moving average forecast
    """
    forecast = []

    # compute the moving average based on the window size
    for time in range(len(series) - window_size):
        forecast.append(series[time:time+window_size])

    # convert tu numpy array
    forecast = np.array(forecast)

    return forecast



if __name__ == '__main__':
    # Generate time steps. Assume 1 per day for one year (365 days)
    time = np.arange(365)

    # Define the slope
    slope = 0.1

    # generate measurements with the defined slope
    series_trend = trend(time, slope)

    # Plot the results
    plot_series(time, series=series_trend, label=[f'slope={slope}'])

    # Generate time steps
    time = np.arange(4 * 365 + 1)

    # Define the parameters of the seasonal data
    period = 365
    amplitude = 40

    #Generate the seasonal data
    series = seasonality(time=time, period=period, amplitude=amplitude)

    # Plot the results
    plot_series(time, series)

    # Define the parameters of the seasonal data
    slope = 0.05
    period = 365
    amplitude = 40

    # Generate the seasonal data
    series = trend(time, slope) + seasonality(time=time, period=period, amplitude=amplitude)

    # Plot the results
    plot_series(time, series)

    ############### Generate Synthetic data ###############
    time = np.arange(4*365+1, dtype="float32")
    baseline = 10
    amplitude = 40
    slope = 0.05
    noise_level = 5

    # create the Series
    series = baseline + trend(time, slope) + seasonality(time, period, amplitude)

    # Update with noise
    series += noise(time, noise_level, seed=42)

    # Plot results
    plot_series(time, series)

    ############## Split The Dataset ###############
    split_time = 1000

    #Get the train set
    time_train  = time[:split_time]
    x_train = series[:split_time]

    # Get the validation set
    time_valid = time[split_time:]
    x_valid = series[split_time:]

    # Plot train set
    plot_series(time_train, x_train)

    # Plot validation set
    plot_series(time_valid, x_valid)


    ########### Naive Forecast ###########
    # Generate the naive forecast
    naive_forecast = series[split_time-1: -1]

    #Define time step
    time_step = 100

    # Print values
    print(f'ground truth at time step {time_step}: {x_valid[time_step]}')
    print(f'The prediction a time step {time_step+1}: {naive_forecast[time_step+1]}')

    # Plot the results
    #plot_series(time_valid, (x_valid, naive_forecast))

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(time_valid, x_valid, 'r-', label='x_valid')
    ax2.plot(time_valid, naive_forecast, 'b-', label='naive_forecast')

    # Plot Zooming in
    #plot_series(time_valid, (x_valid, naive_forecast), start=0, end=150)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(time_valid[0:150], x_valid[0:150], 'g-', label='x_valid')
    ax2.plot(time_valid[0:150], naive_forecast[0:150], 'b-', label='naive_forecast')

    ############ Computing Metrics ###########
    print(keras.metrics.mean_squared_error(x_valid, naive_forecast).numpy())
    print(keras.metrics.mean_absolute_error(x_valid, naive_forecast).numpy)

    ################ Moving Average ###################
    moving_avg = moving_average_forecast(series, window_size=30)[split_time-30: ]

    # Plot resuts
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(time_valid, x_valid, 'r-', label='x_valid')
    ax2.plot(time_valid, moving_avg, 'b-', label='naive_forecast')
