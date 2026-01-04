"""
SYNTHETIC

Utility functions for generating synthetic images.

Author: Peter Lais
Last updated: 10/15/2022
"""

from typing import Collection, Optional, Tuple, Union
import opensimplex as osi
import numpy as np
import neurite as ne
import tensorflow as tf
import sys
from skimage.color import gray2rgb, rgb2hsv, hsv2rgb
from skimage.filters import gaussian
from skimage.transform import rotate
import voxelmorph as vxm

from .utils import remove_singletons, normalize
from .io.io import byte_feature, create_example, generate_tensorspec

def generate_network_input_output(hparam_dicts: Collection[dict],
                                  return_tensorspec: bool = False,
                                  out_dtype: tf.dtypes.DType = tf.float32,
                                  **img_gen_kwargs) -> Collection[Union[Collection[tf.Tensor], Collection[tf.TensorSpec]]]:
    """
    Generate a set of synthetic input/output arguments for a network
    given image generation keywork arguments.

    OUTPUTS ARE CURRENTLY ONLY COMPATIBLE WITH HYPERMORPH SEMISUPERVISED
    MODELS.

    Parameters
    ----------
    hparam_dicts : list of dicts
        A list of dictionary values, with each dictionary containing
        information about how to generate outputs corresponding to a
        SINGLE hyperparameter: `[dict_1, dict2] -> [hp1, hp2] = hparams`
    img_gen_kwargs
        Image generation keyword arguments, passed to `synthimglabel`.

    Returns
    -------
    list
        An output structure compatible with input to registration models.
    """

    # generate [inputs, outputs used for loss]
    (m, f), (s_m, s_f) = synthimglbl(out_dtype=out_dtype, **img_gen_kwargs)
    hparams = []
    for hdict in hparam_dicts:
        temp = {**hdict}
        dist = temp.pop('dist')
        val = dist.rvs(**temp)
        hparams.append(val.item() if val.size == 1 else val)
    hparams = tf.constant(hparams, dtype=out_dtype, name='hparams')
    tensor_collection = [[m, f, hparams, s_m], [s_f]]
    
    if not return_tensorspec: return tensor_collection
    else: return tensor_collection, generate_tensorspec(tensor_collection)

def generate_tfrecord_example_format(hparam_dicts: Collection[dict] = [],
                                     return_tensorspec: bool = False,
                                     **img_gen_kwargs) -> Collection[Union[tf.train.Example, tf.TensorSpec]]:
    """
    A convenience method that immediately serializes the output of
    `generate_network_input_output` to a tf.train.Example for writing to a
    TFRecordDataset.

    Parameters
    ----------
    hparam_dicts : list of dicts
        A list of dictionary values, with each dictionary containing
        information about how to generate outputs corresponding to a
        SINGLE hyperparameter: `[dict_1, dict2] -> [hp1, hp2] = hparams`
    img_gen_kwargs
        Image generation keyword arguments, passed to `synthimglabel`.

    Returns
    -------
    tf.train.Example
        An output structure compatible with input to registration models
        in a format that may be written to a TFRecordDataset easily.
    """

    # Get and format the data
    if not return_tensorspec:
        ((m, f, hparams, s_m), (s_f,)) = generate_network_input_output(hparam_dicts=hparam_dicts,
                                                                       return_tensorspec=False,
                                                                       **img_gen_kwargs)
    else:
        ((m, f, hparams, s_m), (s_f,)), tensorspec = generate_network_input_output(hparam_dicts=hparam_dicts,
                                                                                   return_tensorspec=True,
                                                                                   **img_gen_kwargs)
    m, f, s_m, s_f, hparams = map(lambda x: byte_feature(tf.io.serialize_tensor(x)),
                                  (m, f, s_m, s_f, hparams)) 
    
    # Return the serialized example and any tensorspec info
    serialized_example = create_example(moving_img=m,
                                        fixed_img=f,
                                        fixed_lbl=s_f,
                                        moving_lbl=s_m,
                                        hparams=hparams)
    if not return_tensorspec: return serialized_example
    else: return serialized_example, tensorspec

# BUG: flips non-square dimensions.
def fieldgen(num_fields: int,
             full_shape: Collection[int],
             granularity_factor: Union[float, Collection[float]],
             components: int = 2,
             in_os_seed: int = 0,
             resolution: float = 1,
             min_field_value: float = -1,
             max_field_value: float = 1,
             add_identity: bool = False) -> Collection[Union[np.ndarray, int]]:
    """
    Method for generating a number of two-dimensional fields with varying
    numbers of components. The components argument is present because
    although this is intended to generate deformation fields, it can be repurposed.
    Uses opensimplex's noise functions to quickly make smoothly varying noise.

    Generated fields take the following shape: `[H, W, C]` where
    H is the field's height, W is the field's width, and C is the number of channels.
    For the output to serve as a *deformation* field, C must be 2.

    Parameters
    ----------
    num_fields : int
        Number of noise fields to make.
    full_shape : length-2 tuple
        The two-dimensional shape of the field to create with the shape (H, W).
    granularity_factor : int
        Granuarity factor is a parameter that controls the INVERSE of the
        "granularity" of the field. Smaller factors cause a slower variation of values
        whereas larger factors cause more rapid variations of parameters.
    components : int
        The number of components to include in the deformation field (value of C
        from the example above). Default is 2.
    in_os_seed : int
        The seed to use when generating fields. Default is 0.
    resolution : int
        The resolution to use when generating deformation fields. If resolution is not
        1, then the resultant field will be upscaled (if resolution < 1) or downscaled
        (if resolution > 1) to fit `full_shape`. Default is 1.
    min_field_value : int
        The minimum value of any of the field's components. Default is -1.
    max_field_value : int
        The maximum value of any of the field's components. Default is 1.
    add_identity : int
        Whether to add the identity transform to the deformation field. (The identity
        transform is simply the result of np.indices.) Default is False.
    
    Returns
    -------
    fields : tf.Tensor
        The generated fields.
    seed : int
        The seed passed into the function. Useful for chaining operations of functions
        that use opensimplex since this value is different from the original seed.
    """


    assert len(full_shape) == 2, 'fieldgen only makes 2-D fields.'
    granularity_factor, resolution = remove_singletons(granularity_factor, resolution, out_lens=2)

    # Determine the shape at resolution scale
    scaled_shape = [round(elem * res) for elem, res in zip(full_shape, resolution)]

    # Make the offsets, aka us. Will have identity transform added to it
    # to make it a deformation field.
    us = []
    for _ in range(num_fields):
        current_u = []
        for _ in range(components):
            osi.seed(in_os_seed)
            current_u.append(normalize(osi.noise2array(np.arange(scaled_shape[0]) * granularity_factor[0],
                                                       np.arange(scaled_shape[1]) * granularity_factor[1]),
                                       min_value=min_field_value,
                                       max_value=max_field_value))
            in_os_seed += 1
        
        # Neurite requires def fields of the style [*vol_shape, num_dimensions] so stack at the end here
        current_u = np.stack(current_u, axis=-1)
        us.append(current_u)
    us = np.stack(us, axis=0) # Stack on the first axis to batch. [batch, w, h, c]

    # Perform the interpolation
    if not np.all([r == 1 for r in resolution]):
        us = np.stack([ne.utils.interpn(vol=tf.convert_to_tensor(u),
                                        loc=tf.convert_to_tensor(
                                            np.stack(np.meshgrid(np.linspace(0, scaled_shape[0], full_shape[0]),
                                                                 np.linspace(0, scaled_shape[1], full_shape[1])),
                                                     axis=-1))) for u in us],
                      axis=0)
    
    # Add the identity to the us to form def fields/deffs
    deffs = us
    if add_identity:
        id_ = np.broadcast_to(np.transpose(np.indices(full_shape,
                                                      dtype=int),
                                        axes=tuple(range(1,len(full_shape)+1)) + (0,)),
                            shape=us.shape)
        deffs += id_

    return deffs, in_os_seed

# To consistently get random outputs, need to make rng outside tf function
# and pass it in. Seed this external generator for reproducible performance
def lblgen2d(num_labels: int,
             out_shape: Collection[int],
             padded_shape: Optional[Collection[int]] = None,
             rp: float = 1,
             res_base_map_def_fields: Union[float, Collection[float]] = [1/8, 1/16, 1/32],
             lbl_gf: float = 1,      # label granularity factor
             df_b_gf: float = 1,     # base map granularity factor
             df_o_gf: float = 1,     # out label map granularity factor
             max_l: float = 1,       # max label map value
             min_l: float = -1,      # min label map value
             max_df_b: float = 1,    # max base deformation map value
             min_df_b: float = -1,   # min base deformation map value
             max_df_o: float = 1,    # max output deformation map value
             min_df_o: float = -1,   # min output deformation map value
             df_b_mean: Optional[float] = None,
             df_b_range: Optional[float] = None,
             df_o_mean: Optional[float] = None,
             df_o_range: Optional[float] = None,
             rng: Optional[np.random.Generator] = None,
             seed: Optional[int] = None,
             scale_invariant: bool = False,
             debug: bool = False,
             **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a set of moving/fixed labelmaps.

    This function works by performing the following sequence of events:

    * Generate an UNWARPED label map. This uses `num_labels` opensimplex fields
      to generate a label distribution. This label distribution is then converted
      into a label map based on the argmax of every position down the channel dimension.
    * To introduce more complex deformations, the UNWARPED label map is warped
      using a BASE DEFORMATION FIELD DF_B to form a BASE LABELMAP.
    * Now, to generate two separate but related labelmaps, we need to make two
      SEPARATE DEFORMATION FIELDS DF_O to warp the BASE LABELMAP independently
      to generate two output labelmaps MOVING, FIXED.

    Parameters
    ----------
    num_labels : int
        Number of labels to include in the labelmaps.
    out_shape : length-2 tuple
        The output shape of the labelmaps.
    padded_shape : length-2 tuple
        The internal shape of the image to generate; the excess regions of this
        image are cropped off to reveal outputs of out_shape. This is good for
        preventing regions of black from sneaking in the edges of the image.
        Default is None.
    rp : float
        Resolution at which the deformation fields used to warp the images should
        be generated. Lower resolution leads to more diffuse, slow-varying deformation
        fields. THESE DEFORMATION FIELDS ARE ONLY USED IN GENERATING THE ORIGINAL,
        UNWARPED LABEL MAP. Default is 1.
    res_base_map_def_fields : float or list of floats
        Resolutions at which to generate deformation fields to warp the UNWARPED
        label map to the WAPRED BASE label map. If more than one resolution is
        specified, multiple deformation fields are generated and averaged.
        Default is [1/8, 1/16, 1/32].
    lbl_gf : float
        Labelmap granularity factor. Granuarity factor is a parameter that controls the
        INVERSE of the "granularity" of the field. Smaller factors cause a slower variation of values
        whereas larger factors cause more rapid variations of parameters. Default is 1.      
    df_b_gf : float
        Base deformation field granularity factor; this is the granulariy factor that is
        used when warping the unwarped labelmap to convert it to the BASE labelmap.
        Default is 1.      
    df_o_gf : float
        Granulariy factor to use when warping the BASE labelmap to the MOVING and FIXED
        labelmaps. Default is 1.      
    max_l : float
        Max value of the fields to be used when forming the UNWARPED label map. Default is 1.        
    min_l : float
        Min value of the fields to be used when forming the UNWARPED label map. Default is -1.       
    max_df_b : float
        Max value of the fields to be used when forming the WARPED BASE label map. Default is 1.     
    min_df_b : float
        Min value of the fields to be used when forming the WARPED BASE label map. Default is -1.    
    max_df_o : float
        Max value of the fields to be used when forming the WARPED MOVING/FIXED label map. Default is 1.    
    min_df_o : float
        Min value of the fields to be used when forming the WARPED MOVING/FIXED label map. Default is -1.    
    df_b_mean : float
        If desired, instead specify max and min limits of df_b in terms of mean and range.
        Default is None.
    df_b_range : float
        If desired, instead specify max and min limits of df_b in terms of mean and range.
        Default is None.
    df_o_mean : float
        If desired, instead specify max and min limits of df_o in terms of mean and range.
        Default is None.
    df_o_range : float
        If desired, instead specify max and min limits of df_o in terms of mean and range.
        Default is None.
    rng : np.random.Generator
        Random number generator to use in stochastic operations. Default is None, which
        makes a new generator using `seed`.
    seed : int
        Seed to use if rng is `None`. Default is None.
    scale_invariant : bool
        If scale invariant, normalize the parameters that are dependent on image size 
        (granularity factors, etc.) by the image size. Default is False.
    debug : bool
        If debug mode, return everything as a labeled dictionary. Default is False.

    Returns
    -------
    tuple
        A pair of label maps (moving, fixed).
    """
    
    # Assertions and warnings.
    # Update later.
    #if len(out_shape) != 2: warnings.warn(f"Not tested with "
    #  f"{len(out_shape)}-dimensional generations.")

    # If no padding requested, then set the padded shape equal to the out
    # shape. Else, assert shapes are valid.
    if padded_shape is None:
        padded_shape = out_shape
        no_crop = True
    else:
        no_crop = False
        for i, (pad_dim, out_dim) in enumerate(zip(padded_shape, out_shape)):
            assert pad_dim >= out_dim, 'Padding dimension %d is smaller than the' \
                ' corresponding output dimension: %d vs %d' % (i, pad_dim, out_dim)

    # Initialization of random generators. Note that the default seed
    # is always 0. CHANGE TO MAKING THIS A TUPLE.
    if rng is None: rng = np.random.default_rng(seed)
    if not isinstance(out_shape, np.ndarray): out_shape = np.array(out_shape)
    if not isinstance(padded_shape, np.ndarray): padded_shape = np.array(padded_shape)

    # Seed for opensimplex, for some reason the same seed leads to
    # the same result on ALL generations so must increment this by
    # one for every generation
    os_seed = int(rng.integers(low=0, high=sys.maxsize, size=1))

    # Catch if we define max def field values in terms of range or explicit bounds.
    bound_def = np.any(list(map(lambda x: x is not None,
                                (max_df_b, min_df_b, max_df_o, min_df_o))))
    mr_def = np.any(list(map(lambda x: x is not None,
                             (df_b_mean, df_b_range, df_o_mean, df_o_range))))
    assert not (bound_def and mr_def), 'Deformation field bounds can only be defined in' \
        ' terms of bounds or mean/range.'
    if mr_def:
        # Redefine in terms of maximum and minimum bounds.
        max_df_b = df_b_mean + df_b_range/2
        min_df_b = df_b_mean - df_b_range/2
        max_df_o = df_o_mean + df_o_range/2
        min_df_o = df_o_mean - df_o_range/2

    # Avoid scale invariance unless explicitly specified.
    if scale_invariant:
        # Normalize the granularity factors by max image dimension for consistent performance
        # with upscales in resolution.
        max_dimension = max(padded_shape)
        granularity_factors = [lbl_gf, df_b_gf, df_o_gf]
        granularity_factors = map(lambda x: x / max_dimension, granularity_factors)
        lbl_gf, df_b_gf, df_o_gf = granularity_factors
        
        # Normalize the max/min values by max image dimension for consistent performance
        # with upscales in resolution.
        limits = [max_df_b, min_df_b, max_df_o, min_df_o]
        limits = map(lambda x: x * max_dimension, limits)
        max_df_b, min_df_b, max_df_o, min_df_o = limits   

    # UPDATE WHAT'S GOING ON ABOVE THIS

    # Make a series of random simplex noise arrays at a given resolution,
    # then upscale if desired.
    label_dist, os_seed = fieldgen(num_fields=num_labels,
                                   full_shape=padded_shape,
                                   granularity_factor=lbl_gf,
                                   in_os_seed=os_seed,
                                   resolution=rp,
                                   min_field_value=min_l,
                                   max_field_value=max_l,
                                   components=1)
    label_dist = label_dist[...,0]

    # Make a series of deformation fields. Skip the SVF integration for
    # this step since we have the simplex noise.
    def_fields, os_seed = fieldgen(num_fields=num_labels,
                                   full_shape=padded_shape,
                                   granularity_factor=df_b_gf,
                                   in_os_seed=os_seed,
                                   resolution=rp,
                                   min_field_value=min_df_b,
                                   max_field_value=max_df_b,
                                   add_identity=True)

    # Warp the label distributions by the deformation fields.
    warped_label_dist = np.stack(list(map(ne.utils.interpn,
                                          tf.convert_to_tensor(label_dist),
                                          tf.convert_to_tensor(def_fields))),
                                 axis=0)

    # Collapse the label dist into a single label map.
    base_map = np.argmax(warped_label_dist, axis=0).astype(float)

    # Make deformation fields at different resolutions, stack them, and add them.
    # Do this twice since we need two deformation fields to warp base_map by.
    if '__len__' not in dir(res_base_map_def_fields):
        res_base_map_def_fields = [res_base_map_def_fields]

    # Make the two def fields for labels
    deffs_labels = []
    for _ in range(2):
        deffs = []
        for res in res_base_map_def_fields:
            deff, os_seed = fieldgen(num_fields=1,
                                     full_shape=padded_shape,
                                     granularity_factor=df_o_gf,
                                     in_os_seed=os_seed,
                                     resolution=res,
                                     min_field_value=min_df_o,
                                     max_field_value=max_df_o,
                                     add_identity=True)
            deffs.append(deff[0,...])
        deffs_labels.append(np.mean(deffs, axis=0))
    deffs_labels = np.stack(deffs_labels, axis=0)
    
    out_maps = np.stack(list(map(lambda lbl: ne.utils.interpn(tf.convert_to_tensor(base_map),
                                                              tf.convert_to_tensor(lbl),
                                                              interp_method='nearest'),
                                 deffs_labels)))

    # Crop to the center of the padded dimensions.
    if not no_crop:
        start_r, start_c = map(lambda pad_dim, out_dim: pad_dim // 2 - out_dim // 2,
            padded_shape, out_shape)
        end_r, end_c = map(lambda out_dim, coord: out_dim + coord,
            out_shape, (start_r, start_c))
        base_map = base_map[..., start_r:end_r, start_c:end_c]
        out_maps = out_maps[..., start_r:end_r, start_c:end_c]

    # Return the stacked label maps as a tuple.
    if not debug: return tuple([*out_maps, base_map])
    return {'label_dist': label_dist,
            'base_def_fields': def_fields,
            'warped_label_dist': warped_label_dist,
            'base_map': base_map,
            'out_def_fields': deffs_labels,
            'out_maps': out_maps}

def random_palette(num_colors: int,
                   channels: int,
                   jitter: float,
                   base_color: Optional[Collection[float]] = None,
                   seed: Optional[int] = None,
                   rng: Optional[np.random.Generator] = None,
                   random_bg: bool = False,
                   **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a random palette of colors that may or may not center
    around a base color. If there is no base color, one is randomly
    generated.

    Parameters
    ----------
    num_color : int
        The number of colors to generate.
    channels : int
        The number of channels to generate. (3 for RGB, 4 for RGBA, etc.)
    jitter : float
        The maximum amount by which a single channel's value may change from
        the base color. A primitive means by which to increase the randomness
        of the generated colors; more jitter means more variation in colors.
    base_color : tuple of floats
        A tuple of length `channels` that defines the base color to use.
        Default is None; if None, make a random base color.
    rng : np.random.Generator
        Random number generator to use for stochastic operations. If
        None, make a new generator with `seed`.
    seed : int
        Seed to use for stochastic operations if rng is None and a new
        random generator is made.
    random_bg : bool
        Whether to artifically include a background. If True, one of the colors
        will be replaced with black or white.
    kwargs
        Additional arguments are ignored but included for convenience purposes.
    
    Returns
    -------
    np.ndarray
        An array of shape `(num_colors, channels)` defining the output colormap.
    """

    # Initialize the random generator.
    if rng is None: rng = np.random.default_rng(seed)
    if base_color is None: base_color = rng.uniform(size=(channels,))

    # Generate the colors to be used.
    colors = np.clip(a=rng.uniform(size=(num_colors,channels),
                                   low=-jitter,
                                   high=jitter) + base_color,
                     a_min=0,
                     a_max=1)

    # Randomly assign one of the colors to background if desired.
    if random_bg:
        bg_color = rng.integers(low=0, high=2, size=(channels,))
        idx = rng.integers(low=0, high=num_colors, size=(1,))
        colors[idx, :] = bg_color

    return colors, base_color

## Generate the images themselves by assigning intensities to regions.
# Remove numpy generation.
def corrupt_img(img: np.ndarray,
                stdevs_gaussian_kernel: Union[float, Collection[float]],
                sigma_bias_field: float, 
                rb: float,
                gamma: float,
                seed: Optional[int] = None,
                rng: Optional[np.random.Generator] = None,
                multichannel: bool = False,
                scale_invariant: bool = False,
                **kwargs) -> np.ndarray:
    """
    'Corrupt' an image by adding blur and brightness variation.

    Parameters
    ----------
    img : np.ndarray
        A two-dimensional image to corrupt.
    stdevs_gaussian_kernel : float or list of floats
        A series of floats that specify anisotropic blurring for our image.
        Ideally should include two values in a list format.
    sigma_bias_field : float
        A parameter controlling the standard deviation of the brightness
        variations for the image. Higher sigmas make for more variations
        in the brightness field.
    rb : float
        Resolution at which the bias field is generated. Lower resolutions
        make for more slowly varying bias fields, which are all upsampled
        to the size of the image after generation.
    gamma : float
        A parameter controlling the INTENSITY of variations within the bias
        field. Larger gammas lead to larger variations.
    rng : np.random.Generator
        Random number generator to use for stochastic operations. If
        None, make a new generator with `seed`.
    seed : int
        Seed to use for stochastic operations if rng is None and a new
        random generator is made.
    multichannel : bool
        Whether to treat the image as multichannel or not.
    scale_invariant : bool
        If scale invariant, normalize the parameters that are dependent on image size 
        (granularity factors, etc.) by the image size. Default is False.
    
    Returns
    -------
    corrupt : np.ndarray
        The corrupted image.
    """

    ## Initialization.
    if rng is None: rng = np.random.default_rng(seed)
    full_size = np.array(img.shape)
    if not multichannel:
        if img.shape[-1] != 1: img = np.expand_dims(img, axis=-1)
        img = gray2rgb(img)

    if scale_invariant: stdevs_gaussian_kernel *= max(img.shape[:2])
    stdevs_gaussian_kernel = remove_singletons(stdevs_gaussian_kernel, out_lens=2)
    if '__len__' in dir(stdevs_gaussian_kernel): # if anisotropic, expand blur seq len to image size
        stdevs_gaussian_kernel = list(stdevs_gaussian_kernel)
        stdevs_gaussian_kernel += [1] * (img.ndim - len(stdevs_gaussian_kernel))

    ## Convolve the images to introduce anisotropic blurring.
    blur_img = gaussian(image=img,
                        sigma=stdevs_gaussian_kernel,
                        channel_axis=-1)

    ## Generate an HSV image.
    img_hsv = rgb2hsv(blur_img)

    ## Generate the bias field parameters.
    low_res = np.rint(full_size * rb).astype(int) if not multichannel \
      else np.rint(full_size[:-1] * rb).astype(int)
    b_low_res = np.exp(rng.normal(loc=0,
                                  scale=sigma_bias_field,
                                  size=low_res))
    b_hi_res = ne.utils.interpn(vol=tf.convert_to_tensor(b_low_res),
                                loc=tf.convert_to_tensor(
                                            np.stack(np.meshgrid(np.linspace(0, b_low_res.shape[0], img.shape[0]),
                                                                 np.linspace(0, b_low_res.shape[1], img.shape[1])),
                                                     axis=-1)))

    # Apply the bias field and normalization to each image.
    # Divide the intensity (V) layer by the maximum intensity. We do not
    # min-max normalize since this would be contrast stretching and we don't
    # have negative values here.
    img_hsv[...,-1] *= b_hi_res
    img_hsv[...,-1] = (img_hsv[...,-1] / np.amax(img_hsv[...,-1])) ** np.exp(gamma)

    ## Return the images and label maps.
    return hsv2rgb(img_hsv)

# new, no random support and single use.
def texture_single(patch_shape: Collection[int],
                   vertical_stretch: float,
                   horizontal_stretch: float,
                   rotation: float,
                   clip_mean: float,
                   clip_range: float,
                   opacity_mean: float,
                   opacity_range: float,
                   color1: Collection[float],
                   color2: Collection[float],
                   vertical_stretch_warp: float,
                   horizontal_stretch_warp: float,
                   range_warp: float,
                   mode: str = 'symmetric',
                   scale_invariant: bool = False,
                   **kwargs) -> np.ndarray:
    """
    Generate a textured image patch of a given patch_size.

    Textures consist of a base background with another layer of varying
    opacity being overlaid on it.

    A note on clip_mean/range vs. opacity_mean/range: clip_mean/range
    determines whether the range over which opacity is distributed
    will saturate at low/high values, whereas the opacity_mean/range
    parameters will actually determine the mean and range of the opacities
    on the variable_opacity layer.

    Parameters
    ----------
    patch_shape : length-2 tuple
        Length-2 tuple describing the shape of the patch to be generated.
        Do not include channel information.
    vertical_stretch : float
        The INVERSE of the factor by which the texture should be stretched
        vertically. Smaller values lead to more stretching.
    horizontal_stretch : float
        The INVERSE of the factor by which the texture should be stretched
        horizontally. Smaller values lead to more stretching.
    rotation : float
        Angle of rotation to rotate the texture in degrees of counter-clockwise
        rotation.
    clip_mean :  float
        Range of the interval to which opacity of the overlaid, varying-opacity
        layer will be constrained NORMALIZED FROM 0 TO 1. (See above.)
    clip_range : float
        Range of the interval to which opacity of the overlaid, varying-opacity
        layer will be constrained NORMALIZED FROM 0 TO 1. (See above.)
    opacity_mean : float
        Mean opacity of the varying-opacity layer.
    opacity_range : float
        Range of opacities within the varying-opacity layer.
    color1 : tuple-like
        The color of the constant-color background layer.
    color2 : tuple-like
        The color of the variable-opacity layer.
    vertical_stretch_warp : float
        The INVERSE of the 'granularity' with which texture will be warped in the
        vertical dimension to prevent the appearance of straight, evenly-varying
        textures. (Increases texture complexity.)
    horizontal_stretch_warp : float
        The INVERSE of the 'granularity' with which texture will be warped in the
        horizontal dimension to prevent the appearance of straight, evenly-varying
        textures. (Increases texture complexity.)
    range_warp : float
        The extent to which the texture will be warped. Larger values cause more 
        warp.
    mode : str
        In the rotate method used by skimage, the used strategy to fill in the pixels
        that are out of the domain of the original image. Default is 'symmetric'.
    scale_invariant : bool
        If scale invariant, normalize the parameters that are dependent on image size 
        (granularity factors, etc.) by the image size. Default is False.

    Returns
    -------
    np.ndarray
        The output texture patch.
    """

    # Setup.
    assert len(patch_shape) == 2, "Patches must be 2-dimensional."

    # rename the parameters to be shorter
    params_set = [vertical_stretch, horizontal_stretch, rotation, clip_mean,
                 clip_range, opacity_mean, opacity_range, color1, color2,
                 vertical_stretch_warp, horizontal_stretch_warp, range_warp]
    vs, hs, rot, clipm, clipr, om, ora, c1, c2, vsw, hsw, rw = params_set

    # Scale invariance based on image scale.
    if scale_invariant:
        max_dimension = max(patch_shape)
        vs, hs, vsw, hsw = map(lambda x: x / max_dimension,
                                (vs, hs, vsw, hsw)),
        rw *= max_dimension

    # Opacity array
    opacity = np.expand_dims(rotate(osi.noise2array(np.arange(patch_shape[0]) * vs,
                                                    np.arange(patch_shape[1]) * hs),
                                    rot,
                                    mode=mode), axis=-1)
    opacity = (opacity - opacity.min()) / (opacity.max() - opacity.min())
    opacity = np.clip(opacity, max(0, clipm - clipr/2), max(1, clipm + clipr/2))
    opacity = (opacity - opacity.min()) / (opacity.max() - opacity.min()) * ora + om - ora/2 

    # Make the colors.
    c1, c2 = map(np.asarray, (c1, c2))
    c1, c2 = map(lambda x: x.reshape([1] * (3 - x.ndim) + [-1]), (c1, c2))
    img = np.ones((*patch_shape, 3)) * c1
    img = img * (1 - opacity) + np.ones_like(img) * c2 * opacity

    # Deform the images by a given amount.
    svf = np.stack([osi.noise2array(np.arange(patch_shape[0]) * vsw, np.arange(patch_shape[1]) * hsw),
                    osi.noise2array(np.arange(patch_shape[0]) * vsw, np.arange(patch_shape[1]) * hsw)], axis=-1)
    if svf.min() == svf.max(): svf *= 0
    else: svf = ((svf - svf.min()) / (svf.max() - svf.min()) * 1 - 0.5) * rw
    deff = vxm.layers.VecInt()(svf.reshape(-1, *svf.shape))
    return vxm.layers.SpatialTransformer()([img.reshape(-1, *img.shape), deff])[0].numpy()

def labels2image(labelmaps: np.ndarray,
                 channels: int,
                 num_labels: Optional[int] = None,
                 seed: Optional[int] =None,
                 rng: Optional[np.random.Generator] =None,
                 base_color: Optional[Collection[float]] = None,
                 colors: Optional[np.ndarray] = None,
                 jitter: float = 1,
                 random_bg: bool = False,
                 uniform_colors: bool = False,
                 debug: bool = False,
                 **kwargs) -> np.ndarray:
    """
    Generate a series of images from input labelmaps. Note that there
    is no debug mode necessary since this function does one action.

    Parameters
    ----------
    labelmaps : np.ndarray
        A np.ndarray collection of labelmaps.
    channels : int
        The number of channels to use in the output images.
    num_labels : int
        The number of labels in the labelmaps. Default is None, in which
        case the maximum label present in the labelmaps is inferred to be
        the number of labels.
    rng : np.random.Generator
        Random number generator to use for stochastic operations. If
        None, make a new generator with `seed`.
    seed : int
        Seed to use for stochastic operations if rng is None and a new
        random generator is made.
    base_color : tuple-like 
        A tuple of length `channels` that defines the base color to use.
        Default is None; if None, make a random base color.
    colors : np.ndarray
        An array of shape [num_colors, channels] that can be used to
        supply predefined colors to the function. Default is None, in which
        case new colors are generated.
    jitter : float
        The maximum amount by which a single channel's value may change from
        the base color. A primitive means by which to increase the randomness
        of the generated colors; more jitter means more variation in colors.
        Default is 1.
    random_bg : bool
        Whether to artifically include a background. If True, one of the colors
        will be replaced with black or white.
    uniform_colors : bool
        Whether to include textures in the generated images or to simply
        use uniform colors. If False, includes textures. Default is False.
    kwargs
        Passed to `label2image`, which is the helper for this method.
    
    Returns
    -------
    np.ndarray
        Collection of output images corresponding to the original labelmaps.
    """
  
    # Initialize labels
    if rng is None: rng = np.random.default_rng(seed)
    if num_labels is None: num_labels = int(np.amax(labelmaps[0])) + 1
    if colors is None: colors, _ = random_palette(num_colors=num_labels,
                                                  channels=channels,
                                                  jitter=jitter,
                                                  base_color=base_color,
                                                  seed=seed,
                                                  rng=rng,
                                                  random_bg=random_bg)

    # Make the output images.
    out_imgs = np.stack([label2image(labelmap=labelmap,
                                     channels=channels,
                                     num_labels=num_labels,
                                     seed=seed,
                                     rng=rng,
                                     colors=colors,
                                     jitter=jitter,
                                     uniform_colors=uniform_colors,
                                     random_bg=random_bg,
                                     **kwargs) for labelmap in labelmaps],
                        axis=0)
    
    return out_imgs

def label2image(labelmap: np.ndarray,
                channels: int,
                num_labels: Optional[int] = None,
                seed: Optional[int] = None,
                rng: Optional[np.random.Generator] = None,
                base_color: Optional[Collection[float]] = None,
                colors: Optional[np.ndarray] = None,
                jitter: float = 1,
                random_bg: bool = False,
                uniform_colors: bool = False,
                **kwargs) -> np.ndarray:
    """
    Generate an images from a single input labelmaps.

    Parameters
    ----------
    labelmap : np.ndarray
        A np.ndarray representing a single labelmap.
    channels : int
        The number of channels to use in the output image.
    num_labels : int
        The number of labels in the labelmap. Default is None, in which
        case the maximum label present in the labelmaps is inferred to be
        the number of labels.
    rng : np.random.Generator
        Random number generator to use for stochastic operations. If
        None, make a new generator with `seed`.
    seed : int
        Seed to use for stochastic operations if rng is None and a new
        random generator is made.
    base_color : tuple-like
        A tuple of length `channels` that defines the base color to use.
        Default is None; if None, make a random base color.
    colors : np.ndarray
        An array of shape [num_colors, channels] that can be used to
        supply predefined colors to the function. Default is None, in which
        case new colors are generated.
    jitter : float
        The maximum amount by which a single channel's value may change from
        the base color. A primitive means by which to increase the randomness
        of the generated colors; more jitter means more variation in colors.
        Default is 1.
    random_bg : bool
        Whether to artifically include a background. If True, one of the colors
        will be replaced with black or white.
    uniform_colors : bool
        Whether to include textures in the generated image or to simply
        use uniform colors. If False, includes textures. Default is False.
    kwargs
        Passed to `texture_single`, please refer there for documentation.
    
    Returns
    -------
    np.ndarray
        Collection of output images corresponding to the original labelmaps.
    """

    ## Initialization.

    out_img = np.zeros((*labelmap.shape, channels))
    assert len(out_img.shape) == 3, 'Only single two-dimensional label maps are supported.'
    labelmap_b = np.broadcast_to(np.expand_dims(labelmap, axis=-1), shape=out_img.shape)

    # Initialization
    if rng is None: rng = np.random.default_rng(seed)
    if num_labels is None: num_labels = int(np.amax(labelmap)) + 1
    if colors is None: colors, _ = random_palette(num_colors=num_labels,
                                                  channels=channels,
                                                  jitter=jitter,
                                                  base_color=base_color,
                                                  seed=seed,
                                                  rng=rng,
                                                  random_bg=random_bg)

    for i in range(num_labels):
        # Generate a label map with uniform or varying colors
        central_color = colors[i]
        if not uniform_colors:
            (color1, color2), _ = random_palette(num_colors=2,
                                                 channels=channels,
                                                 jitter=jitter,
                                                 base_color=central_color,
                                                 seed=seed,
                                                 rng=rng,
                                                 random_bg=False)
            texture = texture_single(**kwargs,
                                       patch_shape=out_img.shape[:2],
                                       color1=color1,
                                       color2=color2)
            out_img = tf.where(labelmap_b == i, texture, out_img)
        else: out_img = tf.where(labelmap_b == i, central_color, out_img)
    
    return out_img

# New that supports random variates using scipy-like classes.
def synthimglbl(out_dtype: tf.dtypes.DType = tf.float32,
                **kwargs) -> Union[Tuple[np.ndarray, np.ndarray],
                                   Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]:
    """
    Label generation function that takes in arguments and outputs synthetic
    moving/fixed label/image pairs.

    Each keyword argument may be a direct value or a dictionary of the syntax

    ```
    {
        'dist': scipy.stats distribution
        <other keys for use in dist.rvs>
    }
    ```

    The scipy distribution will be recognized and sampled with the other keys
    in the dictionary being used as keyword arguments. This is a good way
    to introduce stocasticity into the generation outputs. Be sure to seed
    your distributions!

    See `lblgen2d`, `labels2image`, and `corrupt_img` for what to include in
    `kwargs`.

    Parameters
    ----------
    return_labels : bool
        Whether to return labels in the output.
    debug : bool
        If debug, return everything as a dictionary. Default is False.
    kwargs
        Passed to `lblgen2d`, `labels2image`, and `corrupt_img`.


    Returns
    -------
        `(moving_img, fixed_img)` image pair if `return_labels` is False, otherwise
        `((moving_img, fixed_img), (moving_lbl, fixed_lbl))`
    """

    debug = kwargs.get('debug', False)
    assert isinstance(debug, bool), 'Debug must be a boolean variable.'

    # Set up the parameters.
    definite_params = {}
    for name, param in kwargs.items():
        if isinstance(param, dict) and 'dist' in param.keys():
            # If param is a dictionary with specified random distribution,
            # sample it for the fixed parameter. Distribution should
            # adhere to scipy
            temp = {**param}
            dist = temp.pop('dist')
            sample = dist.rvs(**temp)
            definite_params[name] = sample.item() if sample.size == 1 else sample
        else: definite_params[name] = param # otherwise treat parameter as fixed

    if not debug: 
        # Output the images.
        s_m, s_f, _ = lblgen2d(**definite_params)
        m, f = labels2image(labelmaps=tf.stack((s_m, s_f)),
                            **definite_params)
        wm, wf = map(lambda img, gamma: corrupt_img(img=img,
                                                    multichannel=definite_params['channels'] != 1,
                                                    gamma=gamma,
                                                    **definite_params),
                    (m, f),
                    definite_params['gammas'])
        
        # Make all types floating and return.
        s_m, s_f, wm, wf = map(lambda x: tf.cast(x, dtype=out_dtype), (s_m, s_f, wm, wf))
        return (wm, wf) if not definite_params['return_labels'] else ((wm, wf), (s_m, s_f))
    else:
        # Output everything.
        output_dict = lblgen2d(**definite_params)
        m, f = labels2image(labelmaps=tf.convert_to_tensor(output_dict['out_maps']),
                            **definite_params)
        wm, wf = map(lambda img, gamma: corrupt_img(img=img,
                                                    multichannel=definite_params['channels'] != 1,
                                                    gamma=gamma,
                                                    **definite_params),
                    (m, f),
                    definite_params['gammas'])
        output_dict['out_intermed_images'] = tf.stack((m, f))
        output_dict['out_images'] = tf.stack((wm, wf))
        
        # Make all types floating and return.
        output_float_dict = {}
        for k, p in output_dict.items():
            output_float_dict[k] = tf.cast(p, dtype=out_dtype)
        return output_float_dict
