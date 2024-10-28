import numpy as np
from scipy.stats import median_abs_deviation
from scipy.signal import butter, filtfilt
from rich.progress import track

__all__ = ['butterworth_lowpass_filter', 'hampel_filter']

def butterworth_lowpass_filter(data, cutoff, fs, order=5, axis=0):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    
    # Butterworth filter design
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    # Apply the filter along the specified axis
    y = filtfilt(b, a, data, axis=axis)
    
    return y

def hampel_filter(data, window_size=3, n_sigmas=3, axis=0):
    """
    Applies the Hampel filter to a multidimensional array along the specified axis
    and displays progress using a progress bar.
    
    Parameters:
    - data: np.ndarray
        The input data array.
    - window_size: int
        The half-width of the window. The total window size will be 2*window_size + 1.
    - n_sigmas: int
        The number of standard deviations to use for outlier detection.
    - axis: int
        The axis along which to apply the filter.
    
    Returns:
    - filtered_data: np.ndarray
        The array with outliers replaced by the median.
    """
    
    # Moving the axis to be filtered to the first axis
    data = np.moveaxis(data, axis, 0)
    
    filtered_data = np.copy(data)
    total_steps = data.shape[0] - 2 * window_size

    for i in track(range(window_size, data.shape[0] - window_size), description="Applying Hampel Filter"):
        # Define the window
        window = data[i - window_size:i + window_size + 1]
        
        # Compute the median and MAD
        median = np.median(window, axis=0)
        mad = median_abs_deviation(window, axis=0, scale='normal')
        
        # Compute the threshold
        threshold = n_sigmas * mad
        
        # Detect outliers
        difference = np.abs(data[i] - median)
        outliers = difference > threshold
        
        # Replace outliers with the median
        filtered_data[i][outliers] = median[outliers]
    
    # Move the axis back to its original position
    filtered_data = np.moveaxis(filtered_data, 0, axis)
    
    return filtered_data