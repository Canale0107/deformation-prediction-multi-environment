import numpy as np
import cv2
from rich.progress import track

__all__ = ['convert_to_grayscale']

def convert_to_grayscale(images, contrast, brightness):
    """
    グレースケール化、コントラスト調整、明るさ調整を行う
    """
    processed_images = []
    for img in track(images, 'Converting to grayscale...'):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.clip(img + brightness, 0, 255).astype(np.uint8)
        img = np.clip(contrast * img, 0, 255).astype(np.uint8)

        processed_images.append(img)

    return np.array(processed_images)