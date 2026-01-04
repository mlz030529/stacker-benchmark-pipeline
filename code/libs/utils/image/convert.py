"""
UTILS.IMAGE.CONVERT

Functions for interconverting between file formats.

Author: Peter Lais
Last updated: 10/15/2022
"""

import ants
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
import warnings
import tensorflow as tf
from PIL.Image import Image

def plt2img(fig: Figure) -> np.ndarray:
    """
    Convert the passed-in matplotlib figure to a three-channel numpy image.

    Parameters
    ----------
    fig : matplotlib.Figure
        Figure to convert.
    
    Returns
    -------
    np.ndarray
        Converted numpy array.
    """
    # Rasterize the figure
    canvas = FigureCanvas(fig)
    canvas.draw()
    width, height = map(int, fig.get_size_inches() * fig.get_dpi())
    return np.frombuffer(canvas.tostring_rgb(),
                         dtype='uint8').reshape(height, width, 3)

def pil2ants(pil: Image) -> ants.core.ants_image.ANTsImage:
    """
    Convert the passed-in PIL image to ANTs.

    Parameters
    ----------
    pil : PIL Image
        PIL image to convert.
    
    Returns
    -------
    ANTsImage
        Converted ANTs image.
    """
    assert pil.mode in ('1', 'L', 'RGB'), 'Unsupported PIL image mode \'%s\'.' % pil.mode
    np_arr = np.asarray(pil).astype(float)
    return ants.from_numpy(np_arr, is_rgb=(np_arr.ndim == 3 and np_arr.shape[-1] == 3))

def def2img(deff1: np.ndarray, downsample: int = 1, fig_args: dict = {},
            quiver_args: dict = {}) -> tf.Tensor:
    """
    Convert the deformation field deff1 to a three-channel numpy image.

    Parameters
    ----------
    deff1 : np.ndarray
        Deformation field to draw.
    downsample : int
        Number of indices to skip in each direction to decrease the number of arrows drawn.
    fig_args : dict
        Arguments to pass into the creation of the matplotlib Figure.
    quiver_args : dict
        Arguments to pass into the quiver function.
    
    Returns
    -------
    tf.Tensor
        Converted Tensor image of the quiver plot.
    """
    # Make the deformation field
    fig = Figure(**fig_args)
    ax = fig.gca()
    X, Y = np.indices(deff1.shape[:-1])
    U, V = deff1[..., 0], deff1[..., 1]

    # Get the resolution to sample by
    if downsample != 1:
        sample_by_x = deff1.shape[0] // downsample
        sample_by_y = deff1.shape[1] // downsample
        if not sample_by_x:
            warnings.warn('downsample exceeds maximum possible'
                ' in x-direction, setting spacing to 1. Try setting quiver_axis_resolution'
                ' to None.')
            sample_by_x = 1
        if not sample_by_y:
            warnings.warn('downsample exceeds maximum possible'
                ' in y-direction, setting spacing to 1. Try setting quiver_axis_resolution'
                ' to None.')
            sample_by_y = 1
    else: sample_by_x, sample_by_y = 1, 1

    # Make a quiver plot, for some reason ax.axis('square') eliminates all data
    C = quiver_args.pop('C', None)
    if C is not None: ax.quiver(X[::sample_by_x, ::sample_by_y],
                                Y[::sample_by_x, ::sample_by_y],
                                U[::sample_by_x, ::sample_by_y],
                                V[::sample_by_x, ::sample_by_y],
                                C[::sample_by_x, ::sample_by_y],
                                **quiver_args)
    else: ax.quiver(X[::sample_by_x, ::sample_by_y],
                    Y[::sample_by_x, ::sample_by_y],
                    U[::sample_by_x, ::sample_by_y],
                    V[::sample_by_x, ::sample_by_y],
                    **quiver_args)
    
    ax.invert_yaxis()
    ax.axis('equal')
    plt.tight_layout()

    # Rasterize the figure
    return tf.convert_to_tensor(plt2img(fig), dtype=tf.uint8)