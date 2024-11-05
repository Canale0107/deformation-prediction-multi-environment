import tensorflow as tf


__all__ = ['IntersectionOverUnion']


class IntersectionOverUnion(tf.keras.metrics.Mean):
    def __init__(self, threshold=0.5, name='iou', dtype=None):
        super().__init__(name=name, dtype=dtype)
        self.threshold = threshold

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        IoUの状態を更新します。

        Args:
            y_true (tf.Tensor): 実測値（バイナリマスク）。
            y_pred (tf.Tensor): 予測値（バイナリマスク）。
            sample_weight (tf.Tensor, optional): 各サンプルの重み。
        """
        # IoUを計算
        ious = self._calculate_iou(y_true, y_pred)

        # 親クラスの update_state メソッドを呼び出し、状態を更新
        return super().update_state(ious, sample_weight=sample_weight)

    def _calculate_iou(self, y_true, y_pred):
        """
        Calculate IoU for a batch of binary images.

        Args:
            y_true (tf.Tensor): Ground truth binary masks with shape (batch_size, height, width).
            y_pred (tf.Tensor): Predicted binary masks with shape (batch_size, height, width).

        Returns:
            tf.Tensor: IoU scores for each image in the batch.
        """
        # Ensure y_true is cast to float32 for consistency
        y_true = tf.cast(y_true, tf.float32)

        # Binarize predictions based on the threshold
        y_pred_bin = tf.cast(y_pred >= self.threshold, tf.float32)

        # Calculate intersection and union
        intersection = tf.reduce_sum(y_true * y_pred_bin, axis=(1, 2))
        union = tf.reduce_sum(y_true + y_pred_bin, axis=(1, 2)) - intersection

        # Avoid division by zero
        union = tf.maximum(union, tf.keras.backend.epsilon())

        # Calculate IoU for each image in the batch
        return intersection / union