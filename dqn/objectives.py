"""Loss functions."""

import tensorflow as tf
import numpy as np
import keras.backend as K

def huber_loss(y_true, y_pred, max_grad=1.):
    """Calculate the huber loss.

    See https://en.wikipedia.org/wiki/Huber_loss

    Parameters
    ----------
    y_true: np.array, tf.Tensor
      Target value.
    y_pred: np.array, tf.Tensor
      Predicted value.
    max_grad: float, optional
      Positive floating point value. Represents the maximum possible
      gradient magnitude.

    Returns
    -------
    tf.Tensor
      The huber loss.
    """

    with tf.name_scope('HuberLoss'):
        residual = y_true - y_pred

        if max_grad == np.inf:
            huber_loss = K.square(residual) / 2
        else:
            lessthan = K.abs(residual) <= max_grad
            
            sq_loss = K.square(residual) / 2
            abs_loss = max_grad * K.abs(residual) - K.square(max_grad) / 2
            
            huber_loss = tf.where(lessthan, sq_loss, abs_loss)

    return huber_loss


def mean_huber_loss(y_true, y_pred, max_grad=1.):
    """Return mean huber loss.

    Same as huber_loss, but takes the mean over all values in the
    output tensor.

    Parameters
    ----------
    y_true: np.array, tf.Tensor
      Target value.
    y_pred: np.array, tf.Tensor
      Predicted value.
    max_grad: float, optional
      Positive floating point value. Represents the maximum possible
      gradient magnitude.

    Returns
    -------
    tf.Tensor
      The mean huber loss.
    """

    with tf.name_scope('MeanHuberLoss'):
        loss = huber_loss(y_true, y_pred, max_grad=max_grad)
        mean_huber_loss = K.mean(loss)

    return mean_huber_loss
