"""
UTILS.TRANSFORM

Functions to help with transforming both point and image data.

Author: Peter Lais
Last updated: 09/17/2023
"""

from typing import Collection, Literal, Optional, Tuple, Union
from scipy.ndimage import map_coordinates
import SimpleITK as sitk
import numpy as np
import ants
import voxelmorph as vxm
import tensorflow as tf

from .utils import assert_indexing
from .image.coords import points2lblimg, lblimg2points

## INTRODUCED IN V2

def channeltransform(img_ants, transform, **kwargs):
    """
    Transform a (potentially) multichannel ANTs image. If img_ants does 
    have multiple channels, each channel is transformed separately.

    Parameters
    ----------
    img_ants : ANTs image
        ANTs image.
    transform : ANTs transform
        ANTs transform that will be applied over every channel of the image.
    kwargs
        Passed to `ants.apply_transforms`; see this method for details.

    Returns
    -------
    ANTs image
        Registered ANTs image.
    """
    
    # If a single-channel image passed in, then that is the only thing that should
    # be registered.
    img_list = ants.split_channels(img_ants) if img_ants.components > 1 else [img_ants]

    # If string list of transforms, apply these transforms to image.
    # If an actual transform, apply it manually.
    if isinstance(transform, list):
        # If there are no transforms in the supplied list, do not do anything
        if len(transform) == 0:
            return img_ants.copy()
        # Assumes string is not return
        layers = [ants.apply_transforms(channel_img, channel_img, transform, **kwargs) 
            for channel_img in img_list]
    else:
        layers = [transform.apply_to_image(channel_img, **kwargs) 
            for channel_img in img_list]
    
    # Merging an image with one channel produces an identical copy
    return ants.merge_channels(layers) if len(layers) > 1 else layers[0]

## INTRODUCED IN V4

def register_spatial_with_itk_points(spatial_data: np.ndarray,
                                     inverse_itk_trf: sitk.Transform,
                                     spatial_data_indexing: Literal['ij', 'xy'] = 'ij') -> np.ndarray:
    """
    Register spatial data (point data, see how AnnData stores spatial data for input format)
    using an index transform and the ITK transformation package. Takes in spatial data and
    an inverse ITK (or ANTs, as they use this on the backend) transform and warps the
    spatial data according to the forward transform.

    This directly applies an inverse transform to the `spatial_data` points, hence using
    the 'points' strategy.

    Parameters
    ----------
    spatial_data : np.ndarray
        An array of floats identical to the format in which AnnData objects store data/
    inverse_itk_trf : itk.simple.Transform
        An inverse ITK transformation; this may be obtained by calling a transform's
        GetInverse() method.
    spatial_data_indexing : str
        How spatial data are indexed. Default is row-major ('ij') but may be column-major
        ('xy').

    Returns
    -------
    spatial_data_moved : np.ndarray
        Moved spatial data array in the indexing style specified by `spatial_data_indexing`.
    """
    
    # Assertions
    assert spatial_data.ndim == 2 and spatial_data.shape[-1] == 2, \
        'Spatial data must be of the shape (N, 2), got shape %s.' % spatial_data.shape
    spatial_data_indexing = assert_indexing(spatial_data_indexing)
    
    # Convert to numpy
    spatial_data = np.asarray(spatial_data).astype(float)
    
    # Account for indexing: everything is now ij
    if spatial_data_indexing == 'xy': spatial_data = spatial_data[..., ::-1]
    
    # Perform the registration
    spatial_data_moved = apply_itk_trf_points(input=spatial_data,
                                              trf=inverse_itk_trf,
                                              indexing='ij')
    return spatial_data_moved[..., ::-1] if spatial_data_indexing=='xy' else spatial_data_moved

def register_spatial_with_itk_raster(spatial_data: np.ndarray,
                                     inverse_itk_trf: sitk.Transform,
                                     spatial_data_indexing: Literal['ij', 'xy'] = 'ij',
                                     spatial_raster_size: Optional[float] = None,
                                     raster_point_diameter: float = 3) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Register spatial data (point data, see how AnnData stores spatial data for input format)
    using an index transform and the ITK transformation package. Takes in spatial data and
    an inverse ITK (or ANTs, as they use this on the backend) transform and warps the
    spatial data according to the forward transform.

    This converts the spatial data points to a point image, transforms the image, and then
    re-extracts the point data. It is cruder but easier to verify.

    Parameters
    ----------
    spatial_data : np.ndarray
        An array of floats identical to the format in which AnnData objects store data/
    inverse_itk_trf : itk.simple.Transform
        An inverse ITK transformation; this may be obtained by calling a transform's
        GetInverse() method.
    spatial_data_indexing : str
        How spatial data are indexed. Default is row-major ('ij') but may be column-major
        ('xy').
    spatial_raster_size : tuple[int, int]
        Size of the input image in terms of (W, H). This may be obtained from a PIL image
        by simply calling its `size()` method.
    raster_point_diameter : int
        How big to draw the points on the image that is transformed in pixels.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing the moved spatial data array in the indexing style specified by
        `spatial_data_indexing` as its first element and the raster image in numpy format
        as its second element.
    """
    
    # Assertions
    assert spatial_data.ndim == 2 and spatial_data.shape[-1] == 2, \
        'Spatial data must be of the shape (N, 2), got shape %s.' % spatial_data.shape
    spatial_data_indexing = assert_indexing(spatial_data_indexing)
    
    # Convert to numpy
    spatial_data = np.asarray(spatial_data).astype(float)
    
    # Account for indexing: everything is now ij
    if spatial_data_indexing == 'xy': spatial_data = spatial_data[..., ::-1]
    
    spatial_raster = points2lblimg(points=spatial_data,
                                   img_shape=spatial_raster_size[::-1],
                                   point_radius=raster_point_diameter / 2,
                                   indexing='ij')
    spatial_raster = apply_itk_trf_image(input=spatial_raster.astype(float),
                                         trf=inverse_itk_trf.GetInverse(),
                                         interpolator=sitk.sitkNearestNeighbor,
                                         defaultPixelValue=0.0,
                                         outputPixelType=sitk.sitkUnknown,
                                         useNearestNeighborExtrapolator=True)
    spatial_data_moved = lblimg2points(img=spatial_raster.astype(int),
                                       insert_missing=True,
                                       num_points=len(spatial_data),
                                       fill_value=np.nan,
                                       indexing='ij')
    return (spatial_data_moved[..., ::-1] if spatial_data_indexing=='xy' else spatial_data_moved,
            spatial_raster)
        
def register_spatial_with_def_field_points(spatial_data: np.ndarray,
                                           inverse_def_field: np.ndarray,
                                           spatial_data_indexing: Literal['ij', 'xy'] = 'ij',
                                           inverse_def_field_indexing: Literal['ij', 'xy'] ='ij') -> np.ndarray:
    """
    Register spatial data (point data, see how AnnData stores spatial data for input format)
    using an index transform and an inverse deformation field. Takes in spatial data and
    an inverse deformation field and warps the spatial data according to the forward deformation
    field.

    This applies the deformation field directly to the points, hence using the 'points'
    strategy.

    Parameters
    ----------
    spatial_data : np.ndarray
        An array of floats identical to the format in which AnnData objects store data.
    inverse_def_field : np.ndarray or similar
        An inverse deformation field output by a deep learning model.
    spatial_data_indexing : str
        How spatial data are indexed. Default is row-major ('ij') but may be column-major
        ('xy').
    inverse_def_field_indexing : str
        How points in the inverse deformation field should be treated. Default is row-major
        ('ij') but may be column-major ('xy').

    Returns
    -------
    spatial_data_moved : np.ndarray
        Moved spatial data array in the indexing style specified by `spatial_data_indexing`.
    """
    
    # Assertions
    assert spatial_data.ndim == 2 and spatial_data.shape[-1] == 2, \
        'Spatial data must be of the shape (N, 2), got shape %s.' % spatial_data.shape
    spatial_data_indexing, inverse_def_field_indexing = map(assert_indexing,
                                                            (spatial_data_indexing,
                                                             inverse_def_field_indexing))
    assert inverse_def_field.ndim == 3 and inverse_def_field.shape[-1] == 2, \
        'Def field must be of the shape (M, N, 2), got shape %s.' % inverse_def_field.shape
    
    # Convert to numpy
    spatial_data = np.asarray(spatial_data).astype(float)
    inverse_def_field = np.asarray(inverse_def_field).astype(float)
    
    # Account for indexing: everything is now ij
    if spatial_data_indexing == 'xy': spatial_data = spatial_data[..., ::-1]
    if inverse_def_field_indexing == 'xy': inverse_def_field = inverse_def_field[...,::-1]
    
    # Get ready to temporarily move the coordinates to the front to work with map_coordinates.
    _dim_range = np.arange(spatial_data.ndim, dtype=int)
    _dim_range[1:] = _dim_range[:-1]
    _dim_range[0] = spatial_data.ndim - 1

    def _map_coordinates_layerwise(input: np.ndarray, coordinates: np.ndarray):
        """
        A simple helper function for map_coordinates, since map_coordinates can't take
        values down channels.
        """
        resampled = np.zeros(tuple(coordinates.shape[1:]) + (input.shape[-1],), dtype=input.dtype)
        for layer_num in range(input.shape[-1]):
            resampled[..., layer_num] = map_coordinates(input=input[..., layer_num],
                                                        coordinates=coordinates)
        return resampled

    # Perform the registration
    spatial_data_moved = spatial_data + _map_coordinates_layerwise(input=inverse_def_field,
                                                                   coordinates=np.transpose(spatial_data,
                                                                                            _dim_range))
    
    # Eliminated that wicked dependency on tensorflow-addons! The curse is broken!
    # Perform the registration
    # spatial_data_moved = spatial_data + tfa.image.resampler(
    #     data=np.expand_dims(inverse_def_field, axis=0),
    #     warp=np.expand_dims(spatial_data[...,::-1], axis=0))[0].numpy()

    return spatial_data_moved[..., ::-1] if spatial_data_indexing=='xy' else spatial_data_moved

def register_spatial_with_def_field_raster(spatial_data: np.ndarray,
                                           def_field: np.ndarray,
                                           spatial_data_indexing: Literal['ij', 'xy'] = 'ij',
                                           def_field_indexing: Literal['ij', 'xy'] = 'ij',
                                           spatial_raster_size: Optional[float] = None,
                                           raster_point_diameter: float = 3) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Register spatial data (point data, see how AnnData stores spatial data for input format)
    using an index transform and a forward deformation field. Takes in spatial data and
    an inverse deformation field and warps the spatial data according to the forward deformation
    field.

    This convers the spatial data points to an image, applies the deformation field to the image,
    and re-extracts the data points from the transformed image, hence using the 'raster' strategy.

    Parameters
    ----------
    spatial_data : np.ndarray
        An array of floats identical to the format in which AnnData objects store data.
    def_field : np.ndarray or similar
        An deformation field output by a deep learning model.
    spatial_data_indexing : str
        How spatial data are indexed. Default is row-major ('ij') but may be column-major
        ('xy').
    def_field_indexing : str
        How points in the deformation field should be treated. Default is row-major ('ij')
        but may be column-major ('xy').
    spatial_raster_size : tuple[int, int]
        Size of the input image in terms of (W, H). This may be obtained from a PIL image
        by simply calling its `size()` method.
    raster_point_diameter : int
        How big to draw the points on the image that is transformed in pixels.

    Returns
    -------
    spatial_data_moved : np.ndarray
        Moved spatial data array in the indexing style specified by `spatial_data_indexing`.
    """
    
    # Assertions
    assert spatial_data.ndim == 2 and spatial_data.shape[-1] == 2, \
        'Spatial data must be of the shape (N, 2), got shape %s.' % spatial_data.shape
    spatial_data_indexing, def_field_indexing = map(assert_indexing,
                                                            (spatial_data_indexing,
                                                             def_field_indexing))
    assert def_field.ndim == 3 and def_field.shape[-1] == 2, \
        'Def field must be of the shape (M, N, 2), got shape %s.' % def_field.shape
    
    # Convert to numpy
    spatial_data = np.asarray(spatial_data).astype(float)
    def_field = np.asarray(def_field).astype(float)
    
    # Account for indexing: everything is now ij
    if spatial_data_indexing == 'xy': spatial_data = spatial_data[..., ::-1]
    if def_field_indexing == 'xy': def_field = def_field[..., ::-1]
        
    spatial_raster = points2lblimg(points=spatial_data,
                                   img_shape=spatial_raster_size[::-1],
                                   point_radius=raster_point_diameter/2,
                                   indexing='ij').astype(float)
    spatial_raster = vxm.layers.SpatialTransformer(
        interp_method='nearest',
        indexing='ij')([tf.expand_dims(np.stack([spatial_raster] * 3, axis=-1), axis=0),
                        tf.expand_dims(def_field, axis=0)])[0].numpy()
    spatial_raster = np.mean(spatial_raster, axis=-1).astype(int)
    spatial_data_moved = lblimg2points(img=spatial_raster.astype(int),
                                       insert_missing=True,
                                       num_points=len(spatial_data),
                                       fill_value=np.nan,
                                       indexing='ij')
    return (spatial_data_moved[..., ::-1] if spatial_data_indexing=='xy' else spatial_data_moved,
            spatial_raster)


## convert affine ants transform to itk
def ants2itk_affine(trf_ants: ants.core.ants_transform.ANTsTransform) -> sitk.Transform:
    """
    Convert an affine ANTs transform to ITK format without changing
    the action performed by the transform.

    Parameters
    ----------
    trf_ants : ants.core.ants_transform.ANTsTransform
        The transform to convert to ITK format.
    
    Returns
    -------
    trf_itk : sitk.Transform
        The ITK transform.
    """

    trf_itk = sitk.AffineTransform(trf_ants.dimension)
    trf_itk.SetParameters(trf_ants.parameters)
    trf_itk.SetFixedParameters(trf_ants.fixed_parameters)
    return trf_itk

# convert deformation field to itk
# 'image' is a stand-in for the itk version of a displacement field.
def deff2itk(dispf: np.ndarray,
             indexing: Literal['ij', 'xy'] = 'ij') -> sitk.Transform:
    """
    Convert a displacement field to ITK format without changing
    the action performed by the transform.

    Parameters
    ----------
    dispf : np.ndarray
        A displacement field of the shape [*volshape, num_components]
        where num_components is the dimensionality of the transform.
    indexing : str
        The coordinate system of dispf. 'xy' if the displacement field's
        final dimension is arranged as [x, y, ...] (used by ITK) or 'ij'
        if the displacement field's final dimension is arranged as
        [i, j, ...] (used by numpy). Default is 'ij'.
    
    Returns
    -------
    trf_itk : sitk.Transform
        The ITK transform.
    """

    assert indexing.lower() in ('ij', 'xy'), 'Indexing must be either \'ij\' or \'xy\'.'
    skip = -1 if indexing.lower() == 'ij' else 1
    image = sitk.GetImageFromArray(dispf.astype(float)[..., ::skip], isVector=True)
    return sitk.DisplacementFieldTransform(image)

# scale an affine transform for use with moving and fixed images
def scale_affine(trf: Union[sitk.Transform, ants.core.ants_transform.ANTsTransform],
                 orig_shape_2d: Collection[int], new_shape_2d: Collection[int],
                 indexing: Literal['ij', 'xy'] = 'ij') -> Union[sitk.Transform, 
                                                                ants.core.ants_transform.ANTsTransform]:
    """
    Scale a two-dimensional affine transformation originally made
    on orig_shape_2d to new_shape_2d.

    Parameters
    ----------
    trf : sitk.Transform or ants.core.ants_transform.ANTsTransform
        The transform to scale.
    orig_shape_2d : length-2 tuple
        Shape of the original image on which trf was made using the
        indexing system specified by `indexing`.
    new_shape_2d : length-2 tuple
        Shape of the new image on which trf will be applied using the
        indexing system specified by `indexing`.
    indexing : str
        The coordinate system of dispf. 'xy' if the displacement field's
        final dimension is arranged as [x, y, ...] (used by ITK) or 'ij'
        if the displacement field's final dimension is arranged as
        [i, j, ...] (used by numpy). Default is 'ij'. THIS IS UNTESTED;
        STRONGLY ENCOURAGED TO USE 'ij' FOR NOW.
    
    Returns
    -------
    trf_scaled : ants.core.ants_transform.ANTsTransform or sitk.Transform
        The scaled transform.
    """
    
    # handle differently if itk or not
    assert indexing in ('ij', 'xy'), 'Indexing must be either \'ij\' or \'xy\'.'
    skip = 1 if indexing.lower() == 'ij' else -1
    scale = (np.asarray(new_shape_2d) / np.asarray(orig_shape_2d))[::skip]
    if isinstance(trf, sitk.Transform):
        trf = sitk.Transform(trf)
        trf_params = np.array(trf.GetParameters())
        trf_fixed = np.array(trf.GetFixedParameters())
        trf_params[-2:] *= scale
        trf_fixed_scaled = trf_fixed * scale
        trf.SetParameters(tuple(trf_params))
        trf.SetFixedParameters(tuple(trf_fixed_scaled))
        return trf
    elif isinstance(trf, ants.core.ants_transform.ANTsTransform):
        trf_params = trf.parameters
        trf_params[-2:] *= scale
        fixed_parameters_scaled = trf.fixed_parameters * scale
        return ants.create_ants_transform(transform_type=trf.transform_type,
                                          dimension=trf.dimension,
                                          parameters=trf_params,
                                          fixed_parameters=fixed_parameters_scaled)
    else: raise TypeError('Invalid trf type \'%s\'.' % str(type(trf)))

# needs indexing
# to do: trf should become inverse_trf
def apply_itk_trf_image(input: np.ndarray,
                        trf: sitk.Transform,
                        interpolator: int = sitk.sitkLinear,
                        defaultPixelValue: float = 255.0,
                        outputPixelType: int = sitk.sitkUnknown,
                        useNearestNeighborExtrapolator: bool = True,
                        indexing: Literal['ij', 'xy'] = 'ij') -> np.ndarray:
    """
    Apply an ITK transform to an input numpy image. Note that this
    transform is assumed to be an inverse that maps the points of the
    target domain to the points of the source (input) domain since resampling
    is used to obtain the target image.
    
    Due to the above, `trf.GetInverse()` should be used in `apply_itk_trf_points`
    if you want to deform points in the same way as images are being deformed
    in this method.

    'sitk' refers to SimpleITK.

    Parameters
    ----------
    input : np.ndarray
        Original input image of dimensionality (H, W) or (H, W, C).
    trf : sitk.Transform
        The transform to apply to the image during resampling.
    interpolator : int
        The type of interpolation to use when resampling. Default is sitk.sitkLinear.
    defaultPixelValue : float
        The pixel value to use for points outside the domain of the original image.
        Default is 255.0.
    outputPixelType : int
        Output pixel type to use. Default is sitk.sitkUnknown. See sitk.Resample for
        more information about this parameter.
    useNearestNeighborExtrapolator : bool
        Whether to use the nearest point in the input image domain to infer values
        for points that lie outside of the input image domain. Default is True.
    indexing : str
        The coordinate system of input. 'xy' if the image's indices vary
        by first x (column) and then y (row), which is how SimpleITK defines its
        images, or 'ij' if the images subscribe to numpy's format (indices are
        defined by row and then column). Default is 'ij'.
    
    Returns
    -------
    trf_scaled : ants.core.ants_transform.ANTsTransform or sitk.Transform
        The scaled transform.
    """
    
    input = np.asarray(input)
    dtype = input.dtype
    assert input.ndim in (2,3), 'Input image should have the shape [H, W, C] or [H, W].'

    # Account for images with/without channels and 
    axes_perm = (1, 0) if indexing.lower() == 'ij' else (0, 1)
    insert_last = input.ndim == 3 and input.shape[-1] == 1
    if insert_last: input = input[..., 0]
    if input.ndim == 3: axes_perm += (2,)
     
    input_itk = sitk.GetImageFromArray(np.transpose(input.astype(float), axes=axes_perm),
                                       isVector=True)
    affine_itk = sitk.Resample(image1=input_itk,
                               transform=trf,
                               interpolator=interpolator,
                               defaultPixelValue=defaultPixelValue,
                               outputPixelType=outputPixelType,
                               useNearestNeighborExtrapolator=useNearestNeighborExtrapolator)
    output = np.transpose(sitk.GetArrayFromImage(affine_itk), axes=axes_perm)
    if insert_last: output = np.expand_dims(output, axis=-1)
    
    return output.astype(dtype)

def apply_itk_trf_points(input: np.ndarray, trf: sitk.Transform,
                         indexing: Literal['ij', 'xy'] = 'ij') -> np.ndarray:
    """
    Apply an ITK transform to an input numpy image. Note that this
    transform is assumed to be an inverse that maps the points of the
    target domain to the points of the source (input) domain since resampling
    is used to obtain the target image.
    
    Due to the above, `trf.GetInverse()` should be used in `apply_itk_trf_points`
    if you want to deform points in the same way as images are being deformed
    in this method.

    'sitk' refers to SimpleITK.

    Parameters
    ----------
    input : np.ndarray
        A series of points in the shape (num_points, 2).
    trf : sitk.Transform
        The transform to apply to the image during resampling.
    indexing : str
        The coordinate system of input (either 'xy' or 'ij' depending on the
        indexing systems of the points in the input array). Default is 'ij'.
    
    Returns
    -------
    trf_scaled : ants.core.ants_transform.ANTsTransform or sitk.Transform
        The scaled transform.
    """

    assert indexing.lower() in ('ij', 'xy'), 'Indexing must be either \'ij\' or \'xy\'.'
    input = np.asarray(input)
    step = 1 if indexing.lower() == 'ij' else -1
    assert input.ndim == 2 and input.shape[-1] == 2, 'Input should have the shape [N, 2].'
    return np.apply_along_axis(func1d=lambda point: trf.TransformPoint(point[::step])[::step],
                               axis=-1,
                               arr=input)
