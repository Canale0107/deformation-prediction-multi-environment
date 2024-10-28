import numpy as np
from scipy.interpolate import interp1d

__all__ = [
    'linear_interpolate_resampling', 
    'linear_interpolate_downsampling',
    'average_downsampling',
]

def _linear_interpolate(data, original_timestamps, target_timestamps):
    interpolator = interp1d(original_timestamps, data, kind='linear', axis=0, fill_value="extrapolate")
    return interpolator(target_timestamps)

def linear_interpolate_resampling(non_uniform_data, non_uniform_timestamps, start_timestamp, end_timestamp, target_fps):
    uniform_timestamps = np.arange(start_timestamp, end_timestamp, 1 / target_fps)
    return _linear_interpolate(non_uniform_data, non_uniform_timestamps, uniform_timestamps)

def linear_interpolate_downsampling(original_fps, data, target_fps):
    duration = len(data) / original_fps
    original_timestamps = np.linspace(0, duration, len(data))
    target_timestamps = np.linspace(0, duration, int(duration * target_fps))
    return _linear_interpolate(data, original_timestamps, target_timestamps)

def average_downsampling(original_fps, data, target_fps):
    if target_fps >= original_fps:
        raise ValueError("Target FPS must be less than the original FPS for downsampling.")
    
    ratio = original_fps / target_fps
    downsampled_data = []
    
    for i in range(0, len(data), int(ratio)):
        # Collect data points within the current target frame
        segment = data[i:i + int(ratio)]
        # Compute the average of the segment
        downsampled_data.append(np.mean(segment, axis=0))
    
    return np.array(downsampled_data)