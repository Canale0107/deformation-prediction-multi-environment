import numpy as np
import cv2
from rich.progress import track

__all__ = ['normalize_min_max', 'binarize_images']

def normalize_min_max(images):
    """
    Normalize a batch of images to the range [0, 1].

    Parameters:
    images (numpy.ndarray): The input image array of shape (num_images, height, width).

    Returns:
    numpy.ndarray: The normalized image array of the same shape.
    """
    # Flatten the image array to apply normalization across all pixels at once
    min_val = np.min(images)
    max_val = np.max(images)
    
    # Avoid division by zero
    if min_val == max_val:
        return np.zeros_like(images)
    
    normalized_images = (images - min_val) / (max_val - min_val)
    return normalized_images

def binarize_images(images, threshold_value):
    binary_images = (images > threshold_value).astype(np.int32)
    return binary_images