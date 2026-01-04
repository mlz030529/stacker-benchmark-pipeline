"""
UTILS.IMAGE.COORDS

Functions for handling coordinate data and potentially converting them to other forms.

Author: Peter Lais
Last updated: 10/15/2022
"""

from typing import Collection, Union
from PIL import Image
import numpy as np
import neurite as ne
import tensorflow as tf
import pandas as pd
from skimage.draw import disk
from skimage.measure import regionprops_table


def _coords2patch_img(img: Image.Image, coords: Collection[int]) -> np.ndarray:
    """
    Given the passed-in image and rectangle defined by coords, extract a patch
    of img using PIL methods.

    Parameters
    ----------
    img : PIL Image
        Image from which to extract patches.
    coords : length-4 collection of ints
        Coordinates from which to extract patches in the order (upper, left, lower, right).

    Returns
    -------
    np.array
        Extracted image patch.
    """
    # pil backend for coords2patch
    upper, left, lower, right = coords
    if not isinstance(img, Image.Image):
        # image assumed to be an array if not already an image
        img = Image.fromarray(img)
    return np.array(img.crop((left, upper, right, lower)))


def _coords2patch_ne(img: Union[np.ndarray, tf.Tensor], coords: Collection[int],
                     out_patch_shape: Collection[int], interp_method: str = 'linear',
                     fill_value: Union[float, None] = None) -> np.ndarray:
    """
    Given the passed-in image and rectangle defined by coords, extract a patch
    of img using the `neurite` package.

    Parameters
    ----------
    img : array-like
        Array-stype image from which to extract patches.
    coords : length-4 collection of ints
        Coordinates from which to extract patches in the order (upper, left, lower, right).
    out_patch_shape : length-2 collection of ints
        Shape of the output patch in terms of (height, width).
    interp_method : str
        Interpolation method to use in forming the output patch, since the output patch shape
        need not be the same size specified by coords.
    fill_value : float
        The value to fill background pixels.

    Returns
    -------
    np.array
        Extracted image patch.
    """
    # interpolate arr at the locations defined by upper, left, lower, right
    # have the size specified by out_patch+shape
    # Remember this is rc
    upper, left, lower, right = coords
    coords_ne = np.stack(np.meshgrid(
        np.linspace(upper, lower, out_patch_shape[0]),
        np.linspace(left, right, out_patch_shape[1]),
        indexing='ij'
    ), axis=-1)
    out = ne.utils.interpn(
        tf.convert_to_tensor(img),
        tf.convert_to_tensor(coords_ne),
        interp_method=interp_method,
        fill_value=fill_value).numpy()
    return out


def coords2patch(struct: np.ndarray, coords: Collection[int],
                 backend: str = 'pil', **kwargs) -> np.ndarray:
    """
    Get a patch from structure `struct` at coordinates `coords`.

    Parameters
    ----------
    struct : np.ndarray
        Structure from which a patch will be extracted.
    coords : length-4 array
        Coordinates for the patch of the form [upper, left, lower, right].
    backend : str
        Backend to use for cropping. Note that backend 'ne' can be used for
        interpolation, but 'pil' is likely faster.
    out_patch_shape : length-2 array
        With backend 'ne': desired size of the output patch.
    interp_method : str
        With backend 'ne': desired interpolation method.
    fill_value : int or None
        With backend 'ne': fill value for out-of-image locations. With
        backend 'pil', the fill value will always be 0.

    Returns
    -------
    np.ndarray
        Extracted patch from image `img` with necessary padding.
    """
    backend = backend.lower()
    if backend == 'pil':
        return _coords2patch_img(struct, coords, **kwargs)
    elif backend == 'ne':
        return _coords2patch_ne(struct, coords, **kwargs)
    else:
        raise ValueError('Backend \'%s\' not in %s.' %
                         (backend, ['pil', 'ne']))

# all args are numpy


def points2lblimg(points: np.ndarray, img_shape: Collection[int], point_radius: float,
                  indexing: str = 'ij') -> np.ndarray:
    """
    Given some points and desired image shape, draw the points out in physical space
    of the given dimensions.

    Parameters
    ----------
    points : np.ndarray
        A (N, 2) numpy array representing spatial points in two dimensions. This should be
        the format in which data are held in AnnData spatial files.
    img_shape : length-2 tuple
        The shape of the image in terms of (height, width).
    point_radius : float
        The radius of the points to draw on the output image.
    indexing : str
        The indexing of the points in terms of row-major (ij) or column-major (xy) indexing.

    Returns
    -------
    np.ndarray
        An image with points drawn on it according to the above specifications.
    """
    # make circles on an image using skimage.draw
    # since numpy, all arguments are ij
    assert indexing.lower() in (
        'ij', 'xy'), "Indexing must be either 'xy' or 'ij', got '%s'." % indexing.lower()
    assert len(img_shape) > 1 and len(
        img_shape) < 3, 'Desired image must be 2D with optional third channel dimension.'
    if indexing == 'xy':
        points = points[..., ::-1]
    img = np.zeros(img_shape, dtype=int)
    for i, point in enumerate(points):
        rr, cc = disk(center=point,
                      radius=point_radius,
                      shape=img.shape[:2])
        img[rr, cc] = i + 1  # one-based index since background is 0
    return img

# assumes all labels present in image
# one-indexed labels


def lblimg2points(img: np.ndarray, insert_missing: bool = False, num_points: Union[int, None] = None,
                  fill_value: int = np.nan, indexing: str = 'ij') -> np.ndarray:
    """
    Given an image of distinct points, extract the centroids of each point from the image.

    Parameters
    ----------
    img : np.ndarray
        A (H, W) numpy array representing an image containing points. This should be an array
        of integers, with 0 representing background pixels and values above that representing
        points. Each point should have its own label.
    insert_missing : bool
        If gaps are found in the labels, should we insert the missing values?
    num_points : int or None
        If we are inserting missing points, this specifies the number of points that were
        in the original image so that we can be sure that each point is contained in the output.
    fill_value : int
        What value to fill for points that are missing (if insert_missing is enabled).
        The default value is np.nan.
    indexing : str
        The indexing of the points in terms of row-major (ij) or column-major (xy) indexing.

    Returns
    -------
    np.ndarray
        A (N, 2) numpy array representing spatial points in two dimensions. This should be
        the format in which data are held in AnnData spatial files.
    """
    assert indexing.lower() in (
        'ij', 'xy'), "Indexing must be either 'xy' or 'ij', got '%s'." % indexing.lower()
    rprops = pd.DataFrame(regionprops_table(label_image=img,
                                            properties=('label', 'centroid'))).sort_values(by='label')
    # if we need to insert missing points, add -1, -1 dummy points
    if insert_missing:
        max_val = np.amax(rprops['label'].to_numpy())
        if num_points is not None and num_points < max_val:
            raise ValueError('Expected up to %d labels but got %d.' %
                             (num_points, max_val))
        elif num_points is not None:
            max_val = num_points
        needs_insertion = np.arange(1, max_val + 1, dtype=int)
        needs_insertion = needs_insertion[~np.isin(
            needs_insertion, rprops['label'].to_numpy())]
        if needs_insertion.size > 0:
            rprops = pd.concat(objs=[rprops, pd.DataFrame([[lbl, fill_value, fill_value] for lbl in needs_insertion],
                               columns=rprops.columns)],
                               axis=0,
                               ignore_index=True).sort_values(by='label')

    step = 1 if indexing.lower() == 'ij' else -1
    return rprops.loc[:, ['centroid-0', 'centroid-1'][::step]].to_numpy()


def genpatchlocs(img_shape_2d: Collection[int], patch_shape_2d: Collection[int],
                 scale_factor_2d: Union[float, Collection[float]] = 1, overlap: int = 0) -> np.ndarray:
    """
    Given an image shape, desired patch size, scaling, and overlap, generate a series
    of indices that slice the original image shape into a two-dimensional grid of
    overlapping patches.

    Parameters
    ----------
    img_shape_2d : length-2 tuple
        The shape of the overall image expressed in terms of (height, width).
    patch_shape_2d : length-2 tuple
        The shape of the patch expressed in terms of (height, width).
    scale_factor_2d : float or length-2 tuple of floats
        The scaling factor by which to zoom in or out for the patches.
    overlap : int
        The amount in pixels by which each patch should overlap with other patches.

    Returns
    -------
    np.ndarray
        A series of patch locations organized into a two-dimensional array. Each
        element along the 0 dimension indicates a new patch, whereas the first
        dimension is organized into (row_start, col_start, row_end, col_end).
    """
    # Returns patch locations tiled over an image with img_shape_2d

    # Handle all variables.
    def assertion(var, name):
        if not '__len__' in dir(var):
            var = np.array((var, var))
        else:
            var = np.asarray(var)
        assert len(var) == 2, "Length-%d" % len(var) \
            + " %s not compatible with two-dimensional image." % name
        return var
    img_shape_2d, patch_shape_2d, scale_factor_2d, overlap = map(
        assertion,
        (img_shape_2d, patch_shape_2d, scale_factor_2d, overlap),
        ('image shape', 'patch shape', 'scale factor', 'overlap')
    )

    # Handle the location information generation.
    starts_big = np.stack((np.meshgrid(
        np.arange(0, img_shape_2d[0]-patch_shape_2d[0] +
                  1, patch_shape_2d[0]-overlap[0]),
        np.arange(0, img_shape_2d[1]-patch_shape_2d[1] +
                  1, patch_shape_2d[1]-overlap[1]),
        indexing='ij')))  # used to do [::-1] but ij should handle it
    ends_big = starts_big + np.array(patch_shape_2d).reshape(-1, 1, 1)
    # translate to small scale with mult
    loc_info = np.stack((starts_big, ends_big), axis=0) * \
        scale_factor_2d.reshape(1, -1, 1, 1)
    loc_info = np.moveaxis(loc_info, source=(0, 1), destination=(-2, -1))
    loc_info = loc_info.reshape(*loc_info.shape[:2], -1)
    loc_info = np.rint(loc_info).astype(int)  # round off
    return loc_info  # organiced into rstart, cstart, rend, cend
