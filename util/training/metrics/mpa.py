import tensorflow as tf


__all__ = ['MeanPixelAccuracy']


class MeanPixelAccuracy(tf.keras.metrics.Mean):
    """
    Custom metric class to calculate the Mean Pixel Accuracy (mPA) for binary segmentation tasks.
    
    Attributes:
        threshold (float): The threshold value to binarize the predicted values. Default is 0.5.
    
    Methods:
        update_state(y_true, y_pred, sample_weight=None):
            Updates the state of the metric by calculating the mPA between the true and predicted values.
    """

    def __init__(self, threshold=0.5, name='mpa', **kwargs):
        """
        Initializes the MeanPixelAccuracy metric.

        Args:
            threshold (float, optional): The threshold value to binarize the predicted values. Default is 0.5.
            name (str, optional): The name of the metric. Default is 'mpa'.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(name=name, **kwargs)
        self.threshold = threshold

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Updates the metric state with the Mean Pixel Accuracy (mPA) computed from the true and predicted labels.
        
        Args:
            y_true (Tensor): The ground truth binary labels.
            y_pred (Tensor): The predicted labels, typically the output from a model.
            sample_weight (Tensor, optional): Optional sample weights for weighting the metric.
        
        Returns:
            A Tensor representing the updated state of the metric.
        """
        # Binarize predictions based on the threshold
        y_pred_bin = tf.cast(y_pred >= self.threshold, tf.float32)
        
        # Calculate True Positive, True Negative, False Positive, and False Negative
        TP = tf.reduce_sum(tf.cast(y_true == 1, tf.float32) * y_pred_bin, axis=(1, 2))
        TN = tf.reduce_sum(tf.cast(y_true == 0, tf.float32) * tf.cast(y_pred_bin == 0, tf.float32), axis=(1, 2))
        FP = tf.reduce_sum(tf.cast(y_true == 0, tf.float32) * y_pred_bin, axis=(1, 2))
        FN = tf.reduce_sum(tf.cast(y_true == 1, tf.float32) * tf.cast(y_pred_bin == 0, tf.float32), axis=(1, 2))
        
        # Calculate per-class pixel accuracy
        PA_1 = TP / tf.maximum(TP + FN, tf.keras.backend.epsilon())  # Pixel Accuracy for class 1
        PA_0 = TN / tf.maximum(TN + FP, tf.keras.backend.epsilon())  # Pixel Accuracy for class 0
        
        # Calculate mean Pixel Accuracy (mPA)
        mPA = (PA_1 + PA_0) / 2
        
        # Update the state with the calculated mPA
        return super().update_state(mPA, sample_weight=sample_weight)