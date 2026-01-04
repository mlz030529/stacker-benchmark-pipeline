"""
LOSSES

Useful loss functions during model training.

Author: Peter Lais
Last updated: 10/15/2022
"""

import tensorflow as tf
import voxelmorph as vxm

def zero_loss(_: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Simple loss function that returns zero. Good for ignoring values in which
    the loss should never contribute to the output. Preserves gradients
    to remain compatible with backprop.

    Follows the typical parameters of tf.keras.losses.Loss.
    """
    return 0 * tf.math.reduce_sum(y_pred)

def grad_loss(y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight: tf.Tensor=1) -> tf.Tensor:
    """
    Simple loss function that returns the gradient of a deformation field.
    Uses Euclidian distance to calculate the magnitude penalty. Sample
    weights can be passed in to enable hypermorph training.

    Follows the typical parameters of tf.keras.losses.Loss.
    """
    return sample_weight * vxm.losses.Grad(penalty='l2').loss(y_true, y_pred)

def dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Simple loss function that returns the negative Dice score of two
    labelmaps.

    Follows the typical parameters of tf.keras.losses.Loss.
    """
    return vxm.losses.Dice().loss(y_true, y_pred)