"""
EXAMPLE SERIALIZATION/DESERIALIZATION CONSTANTS

Contains example values to pass into various functions in the library. Functions that
these constants may be used on are specified below.

Author: Peter Lais
Last updated: 10/15/2022
"""

import tensorflow as tf

##### CONSTANTS FOR utils.io.load_dict_dataset AND SIMILAR

# Example description to load examples from TFRecords.
EXAMPLE_DESCRIPTION = {'moving_img': tf.io.FixedLenFeature([], tf.string),
                       'fixed_img': tf.io.FixedLenFeature([], tf.string),
                       'fixed_lbl': tf.io.FixedLenFeature([], tf.string),
                       'moving_lbl': tf.io.FixedLenFeature([], tf.string),
                       'hparams': tf.io.FixedLenFeature([], tf.string),}

# Example out types to deserialize TFRecord examples.
EXAMPLE_OUT_TYPES = {'moving_img': tf.float32,
                     'fixed_img': tf.float32,
                     'fixed_lbl': tf.float32,
                     'moving_lbl': tf.float32,
                     'hparams': tf.float32}

# Example output shapes when deserializing TFRecord examples.
EXAMPLE_OUT_SHAPES = {'moving_img': (256,256,3),
                      'fixed_img': (256,256,3),
                      'fixed_lbl': (256,256),
                      'moving_lbl': (256,256),
                      'hparams': (1,)}