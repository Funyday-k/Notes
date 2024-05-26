import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.interpolate import interp1d
import logger

class LightCurvePlot:
    def __init__(self, filename, energy_range, title):
        self.filename = filename
        self.energy_range = energy_range
        self.title = title

    def read_data(self):
        with open(self.filename, 'r') as file:
            for _ in range(5):
                next(file)
            data = []
            for line in file:
                row = [float(x) for x in line.split()]
                data.append(row)
        columns = ['Tstart (s)', 'Counts/s']  # Assuming these are the column names
        self.data = pd.DataFrame(data, columns=columns)

    def detrend_rescale(self):
        self.detrended_data = self.data.copy()
        self.detrended_data['Counts/s'] -= self.detrended_data['Counts/s'].mean()
        self.detrended_data['Counts/s'] /= max(abs(self.detrended_data['Counts/s']))

    def plot_detrended_time_series(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.detrended_data['Tstart (s)'], self.detrended_data['Counts/s'], label='Detrended Counts/s')
        plt.xlabel('Time (s)')
        plt.ylabel('Detrended Counts/s')
        plt.title('Detrended Time Series')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_acf_detrended(self):
        plot_acf(self.detrended_data['Counts/s'], lags=50)
        plt.title('Autocorrelation Function (ACF) of Detrended Series')
        plt.xlabel('Lag')
        plt.ylabel('ACF')
        plt.show()

    def time_series_decomposition(self):
        result = seasonal_decompose(self.detrended_data['Counts/s'], model='additive', period=100)
        result.plot()
        plt.suptitle('Time Series Decomposition')
        plt.show()

    def interpolate(self, method='linear'):
        x = self.data['Tstart (s)']
        y = self.data['Counts/s']
        x_interp = np.linspace(x.min(), x.max(), len(x))
        f = interp1d(x, y, kind=method)
        y_interp = f(x_interp)
        self.interpolated_data = pd.DataFrame({'Tstart (s)': x_interp, 'Counts/s': y_interp})

    def plot_interpolated_time_series(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.data['Tstart (s)'], self.data['Counts/s'], label='Original Counts/s', color='blue')
        plt.plot(self.interpolated_data['Tstart (s)'], self.interpolated_data['Counts/s'], label='Interpolated Counts/s', color='red')
        plt.xlabel('Time (s)')
        plt.ylabel('Counts/s')
        plt.title('Original vs Interpolated Time Series')
        plt.legend()
        plt.grid(True)
        plt.show()

    def cumulative_sum(self):
        self.data['Cumulative Sum'] = self.data['Counts/s'].cumsum()

    def plot_cumulative_sum(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.data['Tstart (s)'], self.data['Cumulative Sum'], label='Cumulative Sum')
        plt.xlabel('Time (s)')
        plt.ylabel('Cumulative Sum')
        plt.title('Cumulative Sum Over Time')
        plt.legend()
        plt.grid(True)
        plt.show()

    def save_transformed_data(self, filename):
        self.data.to_csv(filename, index=False)

class DetrendingAnalyzer(LightCurvePlot):
    def __init__(self, filename, energy_range, title):
        super().__init__(filename, energy_range, title)

    def preprocess_data(self):
        if self.data is not None:
            try:
                self.data['Counts/s'] -= self.data['Counts/s'].mean()
                self.data['Counts/s'] /= max(abs(self.data['Counts/s']))
                self.logger.info("Data detrended and rescaled.")
            except Exception as e:
                self.logger.error(f"Error detrending data: {e}")

    def plot_detrended_time_series(self):
        if self.data is not None:
            plt.figure(figsize=(10, 6))
            plt.plot(self.data['Tstart (s)'], self.data['Counts/s'], label='Detrended Counts/s')
            plt.xlabel('Time (s)')
            plt.ylabel('Detrended Counts/s')
            plt.title('Detrended Time Series')
            plt.legend()
            plt.grid(True)
            plt.show()
            self.logger.info("Detrended time series plotted.")
        else:
            self.logger.warning("No data to plot.")

    def plot_acf_detrended(self):
        if self.data is not None:
            try:
                plot_acf(self.data['Counts/s'], lags=50)
                plt.title('Autocorrelation Function (ACF) of Detrended Series')
                plt.xlabel('Lag')
                plt.ylabel('ACF')
                plt.show()
                self.logger.info("Autocorrelation function of detrended series plotted.")
            except Exception as e:
                self.logger.error(f"Error plotting autocorrelation function: {e}")
        else:
            self.logger.warning("No data to plot.")

    def time_series_decomposition(self):
        if self.data is not None:
            try:
                result = seasonal_decompose(self.data['Counts/s'], model='additive', period=100)
                result.plot()
                plt.suptitle('Time Series Decomposition')
                plt.show()
                self.logger.info("Time series decomposition performed.")
            except Exception as e:
                self.logger.error(f"Error performing time series decomposition: {e}")
        else:
            self.logger.warning("No data to decompose.")
