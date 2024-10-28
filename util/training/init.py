import random
import numpy as np
import tensorflow as tf

__all__ = ['init_device']

def init_device(seed=None, gpu=None):

    # Set the random seed
    if seed is not None:
        random.seed(seed)  # Python's random seed
        np.random.seed(seed)  # NumPy's random seed
        tf.random.set_seed(seed)  # TensorFlow's random seed

    if gpu is not None:
        devices = tf.config.experimental.list_physical_devices('GPU')
        if devices:
            selected_devices = [devices[i] for i in gpu]
            tf.config.set_visible_devices(selected_devices, 'GPU')
            visible_devices = tf.config.get_visible_devices('GPU')
            print("Visible GPU devices:")
            for device in visible_devices:
                print(device)
        else:
            print("No GPU devices available, switching to CPU.")

    # If no GPU specified or no GPU available, force CPU usage
    if gpu is None or not devices:
        tf.config.set_visible_devices([], 'GPU')
        print("Using CPU.")