"""
UTILS.IMAGE

Basic image processing functions.

Author: Peter Lais
Last updated: 10/15/2022
"""

from typing import Collection
from skimage.draw import rectangle_perimeter
import numpy as np
from scipy.signal import get_window
from tqdm import tqdm
from datetime import datetime
from .coords import coords2patch
import tensorflow as tf
import voxelmorph as vxm
from PIL.Image import Image


# Clip the rectangle perimeter.
def _clip_rectangle_perimeter(*, clip_start: Collection[int] = None, clip_end: Collection[int] = None,
                              **rectangle_perimeter_args):
    """
    Perform the rectangle_perimeter method of sklearn but allow the rectangle
    to run off the ends of the image. This is different from the default behavior of
    this function in sklearn.

    Parameters
    ----------
    rectangle_perimeter_args : dict
        Arguments to feed into the rectangle_perimeter function from sklearn.
    clip_start : length-2 tuple or None
        The coordinate (row, column) at which to begin clipping the rectangle.
    clip_end : length-2 tuple or None
        The coordinate (row, column) at which to stop clipping the rectangle.

    Returns
    -------
    length-2 tuple
        rr, cc coordinate values similar in form to those returned by sklearn's
        rectangle_perimeter function.
    """

    # Get the rectangle perimeter and handle arguments
    rr, cc = rectangle_perimeter(**rectangle_perimeter_args)
    if '__len__' not in dir(clip_start):
        clip_start = (clip_start, clip_start)
    if '__len__' not in dir(clip_end):
        clip_end = (clip_end, clip_end)

    # Remap clip_start and clip_end if needed
    start, end = rectangle_perimeter_args['start'], rectangle_perimeter_args['end']
    clip_start = [clip_start[i] if clip_start[i] is not None
                  else start[i] - 1 for i in range(len(clip_start))]
    clip_end = [clip_end[i] if clip_end[i] is not None
                else end[i] + 2 for i in range(len(clip_end))]

    # Get only the rr and cc values that fall within the valid coordinates.
    row_mask = (rr < clip_end[0]) & (rr >= clip_start[0])
    col_mask = (cc < clip_end[1]) & (cc >= clip_start[1])
    final_mask = row_mask & col_mask
    rr, cc = map(lambda arr: arr[final_mask], (rr, cc))
    return rr, cc

# Window an image with a given scipy window function.


def _window_img(img: np.ndarray,
                window_type: str = 'triang',
                window_size: int = 5,
                preserve_top: bool = False,
                preserve_bottom: bool = False,
                preserve_left: bool = False,
                preserve_right: bool = False,
                suppress_corners: bool = True,
                outline: bool = False) -> np.ndarray:
    """
    Given an image, window the ends of the image by a given window function. The window function may
    be any windowing function present in scipy.

    Parameters
    ----------
    img : np.ndarray
        The image to window.
    window_type : str
        The type of window to use. This may be any window function available in scipy.
    window_size : int
        The size of window to use; this means that a border of `window_size` pixels from
        each edge will be affected by the function.
    preserve_top : bool
        Do not apply the filters to the top of the image.
    preserve_bottom : bool
        Do not apply the filters to the bottom of the image.
    preserve_left : bool
        Do not apply the filters to the left of the image.
    preserve_right : bool
        Do not apply the filters to the right of the image.
    suppress_corners : bool
        If images are going to be stitched together in a square-like pattern, this
        ensures that the window functions are applied twice to the corners of the image
        so that the effect of four images overlapping at each image corner is diminshed.
    outline : bool
        Apply an outline to the outside of the image. Good for debugging stitching.

    Returns
    -------
    np.ndarray
        A properly windowed image.
    """
    # This function will always return a copy.

    # If nothing needs to be windowed, skip everything.
    if preserve_bottom and preserve_top and preserve_left and preserve_right:
        return np.copy(img)

    # Averaging window for the function. We only want one end of a symmetric window
    averaging_window = get_window(
        window=window_type,
        Nx=window_size*2,
        fftbins=False
    )[:window_size]
    if outline:
        averaging_window[[0, -1]] = 0  # outline patches with black

    # Clone the original image.
    img_clone = np.copy(img)
    for i in range(2 if suppress_corners else 1):

        # If we're suppressing corner artifacts, apply to left/right
        # and then top/bottom. If not, apply to all edges at once.
        if suppress_corners:
            if i == 0:
                preserve_bottom_temp = True
                preserve_top_temp = True
                preserve_left_temp = preserve_left
                preserve_right_temp = preserve_right
            else:
                preserve_bottom_temp = preserve_bottom
                preserve_top_temp = preserve_top
                preserve_left_temp = True
                preserve_right_temp = True
        else:
            preserve_bottom_temp = preserve_bottom
            preserve_top_temp = preserve_top
            preserve_left_temp = preserve_left
            preserve_right_temp = preserve_right

        # Determine what to window. (If we want to preserve an edge, offset
        # the rectangle outside of the patch.)
        start_offset = np.zeros(2)
        end_offset = np.zeros(2)
        if preserve_bottom_temp:
            end_offset[0] += len(averaging_window)
        if preserve_top_temp:
            start_offset[0] -= len(averaging_window)
        if preserve_right_temp:
            end_offset[1] += len(averaging_window)
        if preserve_left_temp:
            start_offset[1] -= len(averaging_window)

        # Apply the window to the select edges of the image, clipping off
        # coordinates that are offset outside the rectangle.
        end = np.array(img.shape[:2])
        for i, coeff in enumerate(averaging_window):
            start = np.array((i+1, i+1))
            rr, cc = _clip_rectangle_perimeter(start=start+start_offset,
                                               end=end-2-i+end_offset, clip_start=np.zeros(2), clip_end=end)
            img_clone[rr, cc] *= coeff

    return img_clone

# All arguments should be ndarray
# Work with numpy when possible


def stitch(patches,  # [M, N, W, H, C] or [N, W, H, C] when tiled_shape is not None
           tiled_shape=None,  # if not none, assume patches are flattened.
           overlap=0,  # overlap on width and height
           window_type='triang',
           suppress_corners=True,
           multichannel=False,
           order='r',
           verbose=False,
           outline_patches=False):
    """
    Stitch patches of an structure into a full structure.

    Parameters
    ----------
    patches : np.ndarray
        Patches of the structure to be reconstructed. Must be of shape [M, N, O, P, C]
        (an M x N grid of C-channel O x P patches, where M and O are row dimensions
        and N and P are column dimensions) OR of shape [T, O, P, C] (a collection
        of T C-channel O x P patches) when either patch_shape is not None or
        multichannel is False.
    tiled_shape : length-2 array, optional
        A length-2 tuple denoting the M x N grid of tiles (see documentation for
        `patches`). Default is None.
    overlap : int, optional
        The overlap between adjacent tiles. Default is 0.
    window_type : str, optional
        The type of window to use when overlapping tiles. Default is 'triang'.
        Although this may be set to any type of scipy window, it is recommended
        that this remain the default.
    suppress_corners : bool, optional
        Whether to employ an algorithm that eliminates corner artifacts between
        stitched patches. Default is True. While you may change this to False, it
        is not recommended as it will most likely introduce errors in the output.
    multichannel : bool, optional
        Whether the last dimension of the structure denotes channels. Default is False.
    order : str, optional
        A single character in the set {'r', 'c'}. 'r' denotes row-major arrangement
        of tiles whereas 'c' denotes column-major arrangement of tiles. Default is 'r'.
    verbose : bool, optional
        Whether to report progress. Default is False.

    Returns
    -------
    np.ndarray
        The completed stitched structure.
    """

    # Set up the patches.
    assert patches.ndim == 5 or (patches.ndim == 4 and (tiled_shape is not None
                                                        or not multichannel)), \
        "Invalid number of dimensions in 'patches' (%d). Must either be 5-dimensional" \
        " or 4-dimensional with either a non-None tiled_shape or multichannel set to False."

    # Insert another axis for non-multichannel images
    # Must be done before tiled_shape processing which assumes image has channels
    if not multichannel:
        patches = np.expand_dims(patches, axis=-1)

    # Process tiled_shape
    if patches.ndim == 5:
        tiled_shape = np.asarray(patches.shape[:2])
        patches = patches.reshape(-1, *patches.shape[2:])
    else:
        # Make tiled_shape an ndarray, don't copy unless necessary
        tiled_shape = np.asarray(tiled_shape)
        assert len(tiled_shape) == 2, "The length of tiled_shape must be 2."
        # Make sure tiled shape is valid (can hold all patches)
        assert np.prod(tiled_shape) >= patches.shape[0], \
            'Custom tiled_shape %s isn\'t big enough to accommodate %d patches, can handle %d at most.' \
            % (str(tuple(tiled_shape)), patches.shape[0], np.prod(tiled_shape))

    # Confirm the order in which patches should be stitched
    order = order.lower()
    assert order in ('c', 'r'), 'Order must be either column-major (\'c\') or row-major' \
        ' (\'r\').'

    # Get the shape of each patch and the output image shape, factoring in overlap
    patch_shape = patches.shape[1:-1]
    channels = patches.shape[-1]
    out_img_shape = patch_shape * tiled_shape - (tiled_shape - 1) * overlap
    out_img_shape = np.concatenate((out_img_shape, (channels,)), axis=0)

    # A little something for verbosity.
    def verbose_iterable(iterable, verbose=verbose, **tqdm_kwargs):
        return tqdm(iterable, **tqdm_kwargs) if verbose else iterable

    # Start processing patches and adding them to the output image.
    out_img = np.zeros(out_img_shape)
    for i, patch in verbose_iterable(enumerate(patches), total=len(patches),
                                     desc='[%s] Stitching patches' % datetime.now()):
        patch_loc = np.asarray(np.unravel_index(i, shape=tiled_shape)[
                               ::(-1 if order == 'c' else 1)])
        begin_loc = patch_loc * patch_shape - patch_loc * overlap
        start_r, start_c = begin_loc
        end_r, end_c = begin_loc + patch_shape

        # Determine which edges to preserve and then window the image.
        preserve_dict = {}
        if patch_loc[0] == 0:
            preserve_dict['preserve_top'] = True
        if patch_loc[0] == tiled_shape[0] - 1:
            preserve_dict['preserve_bottom'] = True
        if patch_loc[1] == 0:
            preserve_dict['preserve_left'] = True
        if patch_loc[1] == tiled_shape[1] - 1:
            preserve_dict['preserve_right'] = True
        patch_clone = _window_img(
            img=patch,
            window_type=window_type,
            window_size=overlap,
            suppress_corners=suppress_corners,
            outline=outline_patches,
            **preserve_dict
        )

        # Add to the output image.
        out_img[start_r:end_r, start_c:end_c] += patch_clone

    return out_img


# All arguments should be numpy except when you need something else
# formerly patch_alignment_complete
# Works like a dream!
def patchwise_align(img,  # formerly img_pil, 3-dimensional (or 2-dimensional)
                    deff,  # def_field
                    patch_shape,  # length 3
                    overlap,
                    verbose=False,
                    interp_method_img='linear',
                    interp_method_deff='linear',
                    fill_value_img=None,
                    fill_value_deff=0,
                    outline_patches=False):
    """
    Stitch patches of an structure into a full structure.

    Parameters
    ----------
    img : np.ndarray
        A two-dimensional image, optionally with a third channel dimensions.
    deff : np.ndarray
        A large deformation field. This need not be the same two-dimensional
        shape as img; corresponding patches will be taken here using interpolation.
    patch_shape : length-2 or length-3 array
        A length-2 or length-3 tuple denoting the M X N (X C) sizes of patches.
        C denotes the number of channels in the image and will be inferred if
        omitted.
    overlap : int
        The overlap between adjacent tiles.
    verbose : bool, optional
        Whether to report progress. Default is False.
    interp_method : str
        Desired interpolation method. Default is 'linear'.
    fill_value : int or None
        Fill value for out-of-image locations. Default is 0.

    Returns
    -------
    np.ndarray
        The completed registered image.
    """

    # Preprocess the image to be three-dimensional.
    assert img.ndim in (2, 3), \
        "%d-dimensional arguments for img are not supported." % img.ndim
    reflatten = img.ndim == 2
    if reflatten:
        np.expand_dims(img, axis=-1)

    # Preprocess the patch shape to be three-dimensional.
    patch_shape_arr = np.asarray(patch_shape)
    assert len(patch_shape_arr) in (2, 3), \
        "%d-dimensional patch shape not supported." % len(patch_shape)
    if len(patch_shape_arr) == 2:
        patch_shape_arr = np.concatenate((patch_shape_arr, (img.shape[-1],)),
                                         axis=0)

    # Determine the scale between the pillow size and the def field shape
    # Repeat twice for ease of multiplication later.
    img_shape_2d = np.asarray(img.shape[:2])
    scaling_factor = img_shape_2d / deff.shape[:2]
    scaling_factor = np.concatenate((scaling_factor, scaling_factor), axis=0)

    # "Crop" (expand) the image to go evenly into the patch size.
    # Will crop this off later. Use PIL as a backend since this helps coords2patch
    # CURRENTLY: REPLACES WITH BLACK, SHOULD FILL WITH DEFAULT VALUE
    #img_pil = Image.fromarray(img)
    #orig_pil_size = img_pil.size
    #tiled_shape = np.ceil(img_shape_2d / patch_shape_arr[:2]).astype(int)
    #tiled_shape *= patch_shape_arr[:2]
    # img_pil = img_pil.crop((0,0) + tuple(tiled_shape[::-1])) # need to convert rc to xy index
    # i.e. reverse tiled_shape

    # Multiply the deformation field by the scaling factors since if the image is at a
    # different scale than the deformation field, the arrows need to change size
    # Def field is (..., 2) since 2d
    deff_scaled = deff * scaling_factor[:2].reshape(1, 1, -1)

    # Get the starting and ending locations for the patch
    img_shape_2d_padded = np.ceil(
        img_shape_2d / patch_shape_arr[:2]).astype(int)
    img_shape_2d_padded *= patch_shape_arr[:2]
    starts_big = np.stack((np.meshgrid(
        np.arange(
            0, img_shape_2d_padded[0]-patch_shape_arr[0]+1, patch_shape_arr[0]-overlap),
        np.arange(
            0, img_shape_2d_padded[1]-patch_shape_arr[1]+1, patch_shape_arr[1]-overlap),
        indexing='ij')))  # used to do [::-1] but ij should handle it
    ends_big = starts_big + np.array(patch_shape_arr[:2]).reshape(-1, 1, 1)
    loc_info = np.stack((starts_big, ends_big), axis=0)
    loc_info = np.moveaxis(loc_info, source=(0, 1), destination=(-2, -1))
    loc_info = loc_info.reshape(*loc_info.shape[:2], -1)
    # round off, organiced into rstart, cstart, rend, cend
    loc_info = np.rint(loc_info).astype(int)

    def patch_aligner(point):
        # Patchwise aligner that will be passed to numpy's apply_along_axis function

        # Make the image and def field patches.
        img_patch = coords2patch(struct=img,
                                 coords=point,
                                 out_patch_shape=patch_shape_arr[:2],
                                 backend='ne',
                                 interp_method=interp_method_img,
                                 fill_value=fill_value_img)
        def_patch = coords2patch(struct=deff_scaled,
                                 coords=point/scaling_factor,
                                 out_patch_shape=patch_shape_arr[:2],
                                 backend='ne',
                                 interp_method=interp_method_deff,
                                 fill_value=fill_value_deff)
        img_patch, def_patch = map(lambda x: np.expand_dims(x, axis=0).astype(float),
                                   (img_patch, def_patch))

        # Warp the patches using vxm.layers.SpatialTransformer.
        # numpy autoconverted to tensors
        output_padded_patch = vxm.layers.SpatialTransformer(
            interp_method=interp_method_img,
            indexing='ij',
            single_transform=False,
            fill_value=fill_value_img,
            shift_center=True,
            name='patch_aligner_spatial_transform'
        )([img_patch, def_patch])

        # Center-crop the patch and return the patch without the batched dimension.
        output_patch = tf.keras.layers.CenterCrop(
            *patch_shape_arr[:2])(output_padded_patch).numpy()

        # update progress
        if verbose:
            pbar.update(1)

        return output_patch[0]

    # Get the flattened warped patches and then reshape them into
    # the shape specified by patch_shape_arr
    if verbose:
        pbar = tqdm(total=np.prod(loc_info.shape[:-1]),
                    desc='[%s] Warping patches' % datetime.now())
    warped_patches = np.apply_along_axis(
        func1d=patch_aligner,
        axis=-1,
        arr=loc_info
    )
    if verbose:
        pbar.close()

    # Stitch the large image togther using the SAME overlap.
    registered_large_img = stitch(warped_patches,
                                  tiled_shape=None,
                                  overlap=overlap,
                                  multichannel=True,
                                  window_type='triang',
                                  order='r',
                                  suppress_corners=True,
                                  verbose=verbose,
                                  outline_patches=outline_patches)
    if reflatten:
        registered_large_img = registered_large_img[..., 0]

    # Return the cropped-off image array.
    return registered_large_img[:img_shape_2d[0], :img_shape_2d[1]]


def _get_padded_shape(unpadded_shape: Collection[int], patch_shape: Collection[int],
                      overlap: int) -> Collection[int]:
    """
    Get the image shape that would be present if unpadded_shape were padded to evenly fit
    tiles of patch_shape with a given overlap.

    Parameters
    ----------
    unpadded_shape : length-2 collection
        Shape of the form (height, width) that represents the shape of the image that
        is being subdivided into patches.
    patch_shape : length-2 collection
        Shape of the patches in (height, width).
    overlap : int
        Width in pixels of the overlap between adjacent patches.

    Returns
    -------
    length-2 collection
        The shape of the overall image if it were padded to evenly fit patches.
        Returned as (height, width).
    """

    if '__len__' not in dir(overlap):
        overlap = (overlap, overlap)
    assert len(unpadded_shape) == 2, 'Unpadded shape must be length-2.'
    assert len(patch_shape) == 2, 'Patch shape must be length-2.'
    assert len(overlap) == 2, 'Overlap shape must be a scalar or length-2.'
    ups = np.asarray(unpadded_shape)
    pats = np.asarray(patch_shape)
    ov = np.asarray(overlap)
    assert np.all(pats - ov > 0), 'Overlap cannot exceed patch shape in any dimension.'
    actual_patches = 1 + (ups - pats) / (pats - overlap)
    padded_patches = np.ceil(actual_patches)
    padded_shape = patch_shape + (pats - overlap) * (padded_patches - 1)
    return padded_shape.astype(int)

# known issue: boolean images lose the noise around the edge for some reason.
# probably due to linear interpolation; see below


def pad_img(img: np.ndarray, patch_shape: Collection[int], overlap: int, origin: str = 'center',
            **kwargs) -> np.ndarray:
    """
    Pad an image to evenly fit patches of shape patch_shape featuring a given
    overlap between them.

    For example:

    ```
    >>> img = np.zeros((60,60))
    >>> patch_shape = (16,16)
    >>> overlap = 0
    >>> pad_img(img, patch_shape, overlap).shape
    <<< (64, 64)
    ```

    Parameters
    ----------
    img : np.ndarray
        Image to be padded. Should be a numpy array.
    patch_shape : Collection[int]
        Shape of the patches in (height, width).
    overlap : int
        Width in pixels of the overlap between adjacent patches.
    origin : str
        Where the original image should be located in the padded image. Valid options
        are top-left ('topleft', 'tl') or center ('center', 'c').
    
    Returns
    -------
    np.ndarray
        The appropriately padded numpy image.
    """

    img = np.asarray(img)
    padded_shape = _get_padded_shape(unpadded_shape=img.shape[:2],
                                     patch_shape=patch_shape,
                                     overlap=overlap)
    padding_rc = padded_shape - img.shape[:2]

    if origin.lower() in ('center', 'c'):
        start = -padding_rc // 2
    elif origin.lower() in ('topleft', 'tl'):
        start = np.zeros(2)
    else:
        raise ValueError("Invalid origin '%s'." % origin)

    point = start.tolist() + (start + padded_shape).tolist()
    dtype = img.dtype
    img_padded = coords2patch(struct=img.astype(float),
                              coords=point,
                              backend='ne',
                              out_patch_shape=padded_shape,
                              **kwargs)
    return img_padded.astype(dtype)


def unpad_img(img: np.ndarray, orig_shape: Collection[int], origin: str = 'center') -> np.ndarray:
    """
    Remove padding on the passed-in image to get it back to its original shape.

    Parameters
    ----------
    img : np.ndarray
        Image to be unpadded, should be numpy.
    orig_shape : Collection[int]
        The shape of the original unpadded image of the form (height, width).
    origin : str
        Where the original image is positioned within the padded image. Valid options
        are top-left ('topleft', 'tl') or center ('center', 'c').
    
    Returns
    -------
    np.ndarray
        The appropriately unpadded numpy image.
    """

    os = np.asarray(orig_shape)
    if origin.lower() in ('center', 'c'):
        center = np.asarray(img.shape[:2]) // 2
        tl = center - os // 2
        br = tl + os
        return img[tl[0]:br[0], tl[1]:br[1]]
    elif origin.lower() in ('topleft', 'tl'):
        return img[:orig_shape[0], :orig_shape[1]]
    else:
        raise ValueError("Invalid origin '%s'." % origin)

# shape should be 2d


def pad_to_shape(img: np.ndarray, shape_2d: Collection[int], origin: str = 'center',
                 val: float = 0) -> np.ndarray:
    """
    Pad an image to a desired shape.

    Parameters
    ----------
    img : np.ndarray
        Image to be padded. Should be a numpy array.
    shape_2d : Collection[int]
        Shape that the padded image should take in terms of (height, width).
    origin : str
        Where the original image should be located in the padded image. Valid options
        are top-left ('topleft', 'tl') or center ('center', 'c').
    val : float
        Value to use in the padded regions of the image.

    Returns
    -------
    np.ndarray
        The appropriately padded image.
    """
    img = np.asarray(img)
    to_pad = np.asarray(shape_2d) - np.asarray(img.shape[:2])
    assert np.all(
        to_pad >= 0), 'Desired patch size is smaller than image shape.'

    if origin.lower() in ('center', 'c'):
        padding = [(to_pad[0] // 2, to_pad[0] - to_pad[0] // 2),
                   (to_pad[1] // 2, to_pad[1] - to_pad[1] // 2)]
    elif origin.lower() in ('topleft', 'tl'):
        padding = list(to_pad)
    else:
        raise ValueError("Invalid origin '%s'." % origin)

    padding += [(0, 0)] * (img.ndim - 2)
    return np.pad(img, padding, constant_values=val)

# resize a 2d image to a maximum dimension


def resize_to_max_dim_pil(pil: Image, max_dim: int, **kwargs) -> Image:
    """
    Resize a PIL image such that its maximum dimension is equal to max_dim.

    Parameters
    ----------
    pil : PIL Image
        PIL image to resize.
    max_dim : int
        The maximum dimension of the new image.
    
    Returns
    -------
    PIL Image
        The appropriately resized image.
    """
    return pil.resize([round(dim / (np.amax(pil.size[:2]) / max_dim)) for dim in pil.size], **kwargs)
