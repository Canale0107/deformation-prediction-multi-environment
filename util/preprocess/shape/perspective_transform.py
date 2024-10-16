import numpy as np
import cv2
from rich.progress import track

__all__ = ['apply_perspective_transform']

def _sort_corners(corners):
    # 4つの点を順序付けするための関数
    ordered_corners = np.zeros((4, 2), dtype="float32")

    # 左上の点を見つける
    sum_values = corners.sum(axis=1)
    ordered_corners[0] = corners[np.argmin(sum_values)]
    ordered_corners[2] = corners[np.argmax(sum_values)]

    # 左下の点と右上の点を見つける
    diff_values = np.diff(corners, axis=1)
    ordered_corners[1] = corners[np.argmin(diff_values)]
    ordered_corners[3] = corners[np.argmax(diff_values)]

    return ordered_corners

def _compute_perspective_matrix(corners, target_width, target_height):
    # 指定した4点を順序付けする
    ordered_corners = _sort_corners(corners)
    (top_left, top_right, bottom_right, bottom_left) = ordered_corners

    # 変換後の座標を定義
    destination_points = np.array([
        [0, 0],
        [target_width - 1, 0],
        [target_width - 1, target_height - 1],
        [0, target_height - 1]], dtype="float32")

    # 射影変換行列を計算
    transform_matrix = cv2.getPerspectiveTransform(ordered_corners, destination_points)

    return transform_matrix

def apply_perspective_transform(images, corners, target_width, target_height):
    corners_array = np.array(corners)
    transform_matrix = _compute_perspective_matrix(corners_array, target_width, target_height)
    
    transformed_images = []
    for img in track(images, 'Applying perspective transform...'):
        transformed_image = cv2.warpPerspective(img, transform_matrix, (target_width, target_height))
        transformed_images.append(transformed_image)
    return np.array(transformed_images)