import random
import numpy as np
import tensorflow as tf

__all__ = ['init_device']

def init_device(seed=None, gpu=None):
    # Set the random seed
    if seed is not None:
        print(f"Setting random seed: {seed}")
        random.seed(seed)  # Python's random seed
        np.random.seed(seed)  # NumPy's random seed
        tf.random.set_seed(seed)  # TensorFlow's random seed
        print("Random seed set for Python, NumPy, and TensorFlow.")

    # Check for GPUs and set visible devices
    devices = tf.config.experimental.list_physical_devices('GPU')
    if devices:
        print(f"Detected {len(devices)} GPU(s): {[device.name for device in devices]}")
        if gpu is not None:
            try:
                selected_devices = [devices[i] for i in gpu]
                tf.config.set_visible_devices(selected_devices, 'GPU')
                visible_devices = tf.config.get_visible_devices('GPU')
                print(f"Visible GPU devices ({len(visible_devices)}): {[device.name for device in visible_devices]}")
            except IndexError as e:
                print(f"Error: Specified GPU indices {gpu} are out of range for available devices.")
                print("Switching to CPU as fallback.")
                tf.config.set_visible_devices([], 'GPU')
        else:
            print("No specific GPU indices provided, making all GPUs visible.")
            tf.config.set_visible_devices(devices, 'GPU')
    else:
        print("No GPU devices available. Switching to CPU.")

    # If no GPU specified or no GPU available, force CPU usage
    if gpu is None or not devices:
        tf.config.set_visible_devices([], 'GPU')
        print("Using CPU only as no GPUs are available or specified.")