import main
import logging

# System MacOS Python Vesion: 3.9.6
# matplotlib                3.7.0
# matplotlib-inline         0.1.6
# numpy                     1.23.4
# pandas                    2.2.2
# scipy                     1.10.0
# statsmodels               0.14.2

# Example usage:
filename = "data.txt"
energy_range = "50-300 KeV"
title = "Light Curve for Fermi Event BN081224887"  

# Set up logging
logging.basicConfig(level=logging.INFO)

# Create instances of subclasses

detrending_analyzer = main.DetrendingAnalyzer(filename, energy_range, title)
detrending_analyzer.read_data()

detrending_analyzer.detrend_rescale()

detrending_analyzer.plot_detrended_time_series()

detrending_analyzer.plot_acf_detrended()

detrending_analyzer.time_series_decomposition()

detrending_analyzer.display_data()

# Create instances of the InterpolationAnalyzer and CumulativeSummationAnalyzer subclasses
interpolation_analyzer = main.InterpolationAnalyzer(filename, energy_range, title)
cumulative_summation_analyzer = main.CumulativeSummationAnalyzer(filename, energy_range, title)

# Read data
interpolation_analyzer.read_data()
cumulative_summation_analyzer.read_data()

# Interpolate data using linear interpolation
interpolation_analyzer.interpolate_data(method='linear')

# Calculate cumulative sum
cumulative_summation_analyzer.calculate_cumulative_sum()

# Plot interpolated time series
interpolation_analyzer.plot_interpolated_time_series()

# Plot cumulative sum
cumulative_summation_analyzer.plot_cumulative_sum()

#Generate log file

