import tensorflow as tf
from tensorflow.keras.losses import Loss

__all__ = ['IoULoss']

class IoULoss(Loss):
    def call(self, y_true, y_pred):
        y_pred = tf.nn.sigmoid(y_pred)  # Sigmoidを適用する場合
        y_true = tf.cast(y_true, tf.float32)
        
        intersection = tf.reduce_sum(y_true * y_pred)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
        IoU = (intersection + 1e-6) / (union + 1e-6)
        
        return 1 - IoU