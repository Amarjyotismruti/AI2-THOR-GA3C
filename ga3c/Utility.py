import os
import tensorflow as tf
import numpy as np
import keras.backend as K

def get_output_folder(parent_dir, env_name):
    """Return save folder.

    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.

    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.

    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    try:
        os.makedirs(parent_dir)
    except Exception:
        pass

    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    return parent_dir

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