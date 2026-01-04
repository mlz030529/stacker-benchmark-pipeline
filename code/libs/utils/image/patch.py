"""
UTILS.IMAGE.PATCH

Functions for doing basic manipulations for patches. This is better for proof of concept
testing and should not be used by end users.

Author: Peter Lais
Last updated: 10/15/2022
"""

from typing import Collection, Union
import numpy as np

def pad_to_square_patch(img: np.ndarray,
                        constant_values: Union[int, Collection[int]]) -> np.ndarray:
    """
    Zero-pad an image such that its width and height dimensions are equal. The image
    will be padded to the largest dimension.

    Parameters
    ----------
    img : np.ndarray
        The image to zero-pad to a square patch size.
    constant_values : int or length-2 sequence of ints
        Values to pad along each dimension in the same style that would be accepted by
        np.pad. See the reference for `np.pad` for more information.

    Returns
    -------
    np.ndarray
        Image zero-padded to a square shape.
    """
    img = np.asarray(img)
    max_dim = np.amax(img.shape[:2])
    rows_pad = (max_dim - img.shape[0])
    cols_pad = (max_dim - img.shape[1])
    return np.pad(img, [(rows_pad // 2, rows_pad - rows_pad // 2),
                        (cols_pad // 2, cols_pad - cols_pad // 2)] + [(0,0)] * (img.ndim - 2),
                  constant_values=constant_values)
