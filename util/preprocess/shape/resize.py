import numpy as np
import cv2
from rich.progress import track

__all__ = ['resize_images']

def resize_images(images, new_width, new_height):
    """
    画像縮小処理
    """
    resized_images = []
    for img in track(images, 'Resizing images...'):
        # 画像を縮小する
        # cv2.INTER_AREAを使用することで、画像を縮小するために最適化されたアルゴリズムが使用される
        resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        resized_images.append(resized_img)

    return np.array(resized_images)