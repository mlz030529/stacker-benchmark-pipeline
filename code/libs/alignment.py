"""
ALIGNMENT

High-level alignment functions that are to be used by end users.

Author: Peter Lais
Last updated: 10/15/2022
"""

from typing import Callable, Tuple
import numpy as np
from PIL import Image
import voxelmorph as vxm
import tensorflow as tf
import ants
import SimpleITK as sitk

from .utils.image import (pad_img as pad_img_to_even_tile,
                          unpad_img as unpad_img_from_even_tile,
                          stitch,
                          resize_to_max_dim_pil)
from .utils.image.convert import pil2ants
from .utils.image.coords import coords2patch, genpatchlocs
from .utils.image.tissue import istissue, standard_mask_fn
from .models import EnrichedFunctionalModel
from .utils.transform import (scale_affine,
                              ants2itk_affine,
                              apply_itk_trf_image,
                              channeltransform,
                              register_spatial_with_itk_points,
                              register_spatial_with_itk_raster,
                              register_spatial_with_def_field_points,
                              register_spatial_with_def_field_raster)

# Align PIL images using ANTs protocol
def ants_align_pil(mov_pil: Image.Image,
                   fix_pil: Image.Image,
                   mov_mask_pil: Image.Image,
                   defaultvalue: float = 0,
                   type_of_transform: str ='Affine',
                   **kwargs) -> Tuple[Image.Image, Image.Image, dict]:
    """
    Align two PIL images and associated masks using an associated method from
    ANTs. Also intakes a binary image mask from PIL to align. Note that all
    images and masks must be of the same size. Uses ANTs as backend.

    Parameters
    ----------
    mov_pil : PIL.Image
        The moving image to be deformed during alignment.
    fix_pil : PIL.Image
        The fixed image to serve as a reference (remain constant) during alignment.
    defaultvalue : int
        If the image is aligned such that the the new image contains points from outside
        the domain of the original image, this is the value that will be inserted as
        a filler. Default is `0`; if white images, use `255` if RGB.
    **kwargs
        Additional arguments passed to `channeltransform`; see its documentation.
    
    Returns
    -------
    moved_pil : PIL.Image.Image
        The deformed moving image.
    moved_mask_pil : PIL.Image.Image
        The deformed moving image mask.
    affine_tfm_info : dict
        Information supplied by ANTs about the forward transform (used to transform the
        moving to the fixed image) as well as the reverse transform. See
        `[ants.registration](https://antspy.readthedocs.io/en/latest/registration.html)`
        for documentation on this output.
    """
    
    assert np.all([isinstance(pil, Image.Image) for pil in [mov_pil, fix_pil, mov_mask_pil]]), \
        'All images must be of type PIL.Image.Image.'
    assert np.all([np.array_equal(pil.size, mov_pil.size) for pil in [fix_pil, mov_mask_pil]]), \
        'All images must be of equal size.'
    
    # Perform the registration. If the images are both empty, this will emit a warning
    # since it uses undefined transforms.
    tfm_info = ants.registration(fixed=pil2ants(fix_pil.convert('L')),
                                 moving=pil2ants(mov_pil.convert('L')),
                                 type_of_transform=type_of_transform)

    # channeltransform has been updated to handle empty transform lists (just return a copy
    # of the input image).
    mov_aff, mov_mask_aff = map(lambda img, dv, dtype: Image.fromarray(
                                                       channeltransform(img_ants=pil2ants(img),
                                                                        transform=tfm_info['fwdtransforms'],
                                                                        defaultvalue=dv,
                                                                        **kwargs).numpy().astype(dtype)),
                                (mov_pil, mov_mask_pil),
                                (float(defaultvalue), 0),
                                (np.uint8, bool))

    return mov_aff, mov_mask_aff, tfm_info

# Align PIL images using ANTs protocol with ants_initializer
def ants_align_pil2(mov_pil: Image.Image,
                   fix_pil: Image.Image,
                   mov_mask_pil: Image.Image,
                   defaultvalue: float = 0,
                   type_of_transform: str ='Affine',
                   search_factor=20,
                   use_principal_axis=False,
                   **kwargs) -> Tuple[Image.Image, Image.Image, dict]:
    """
    Align two PIL images and associated masks using an associated method from
    ANTs. Also intakes a binary image mask from PIL to align. Note that all
    images and masks must be of the same size. Uses ANTs as backend.

    Parameters
    ----------
    mov_pil : PIL.Image
        The moving image to be deformed during alignment.
    fix_pil : PIL.Image
        The fixed image to serve as a reference (remain constant) during alignment.
    defaultvalue : int
        If the image is aligned such that the the new image contains points from outside
        the domain of the original image, this is the value that will be inserted as
        a filler. Default is `0`; if white images, use `255` if RGB.
    **kwargs
        Additional arguments passed to `channeltransform`; see its documentation.
    
    Returns
    -------
    moved_pil : PIL.Image.Image
        The deformed moving image.
    moved_mask_pil : PIL.Image.Image
        The deformed moving image mask.
    affine_tfm_info : dict
        Information supplied by ANTs about the forward transform (used to transform the
        moving to the fixed image) as well as the reverse transform. See
        `[ants.registration](https://antspy.readthedocs.io/en/latest/registration.html)`
        for documentation on this output.
    """
    
    assert np.all([isinstance(pil, Image.Image) for pil in [mov_pil, fix_pil, mov_mask_pil]]), \
        'All images must be of type PIL.Image.Image.'
    assert np.all([np.array_equal(pil.size, mov_pil.size) for pil in [fix_pil, mov_mask_pil]]), \
        'All images must be of equal size.'
    
    # Perform the registration. If the images are both empty, this will emit a warning
    # since it uses undefined transforms.
    fixedants=pil2ants(fix_pil.convert('L'))
    moveants=pil2ants(mov_pil.convert('L'))
    
    txfile = ants.affine_initializer( fixedants, moveants,
                                 use_principal_axis=use_principal_axis,
                                 search_factor=search_factor,  #may be relaxed if major axis is available.
                                 radian_fraction=1, local_search_iterations=10)
    movants_ini=ants.apply_transforms(fixed=fixedants,
                                  moving= moveants,
                                  interpolator='linear', #nearestNeighbor
                                  transformlist=txfile)

    tfm_info = ants.registration(fixed=fixedants,
                                 moving=movants_ini,
                                 type_of_transform=type_of_transform
                                 #aff_iterations=(21000, 21000, 21000, 21000),
                                 #grad_step=0.2
                                 )
    tx1 = ants.read_transform(txfile)
    txmat1=np.array(tx1.parameters).reshape(3,2).transpose()
    txmat1=np.vstack([txmat1,[0,0,1]])
    #print(txmat1)

    tx2 = ants.read_transform(tfm_info['fwdtransforms'][0])
    #print(tx2.parameters)
    txmat2=np.array(tx2.parameters).reshape(3,2).transpose()
    txmat2=np.vstack([txmat2,[0,0,1]])

    comboparam=(np.matmul(txmat2, txmat1)[0:2,]).flatten(order='F')
    combotx= ants.create_ants_transform(transform_type=tx1.transform_type,
                                          dimension=tx1.dimension,
                                          parameters=comboparam,
                                          fixed_parameters=tx1.fixed_parameters)
    invcombotx=combotx.invert()
    
    warpedmovout=combotx.apply_to_image(moveants)
    warpedfixout=invcombotx.apply_to_image(fixedants)

    fwdfile=tfm_info['fwdtransforms'][0].replace('.mat','_f.mat')
    invfile=tfm_info['invtransforms'][0].replace('.mat','_i.mat')
    ants.write_transform(combotx, fwdfile)
    ants.write_transform(invcombotx, invfile)

    tfm_info['warpedmovout']=warpedmovout
    tfm_info['warpedfixout']=warpedfixout
    tfm_info['fwdtransforms']=[fwdfile]
    tfm_info['invtransforms']=[invfile]
    # channeltransform has been updated to handle empty transform lists (just return a copy
    # of the input image).
    mov_aff, mov_mask_aff = map(lambda img, dv, dtype: Image.fromarray(
                                                       channeltransform(img_ants=pil2ants(img),
                                                                        transform=tfm_info['fwdtransforms'],
                                                                        defaultvalue=dv,
                                                                        **kwargs).numpy().astype(dtype)),
                                (mov_pil, mov_mask_pil),
                                (float(defaultvalue), 0),
                                (np.uint8, bool))

    return mov_aff, mov_mask_aff, tfm_info

# Align PIL images using ANTs protocol
# deprecated, unnecessary.
def ants_align_pil3(mov_pil: Image.Image,
                   fix_pil: Image.Image,
                   mov_mask_pil: Image.Image,
                   defaultvalue: float = 0,
                   type_of_transform: str ='Affine',
                   use_principal_axis=True,
                   search_factor=20,
                   **kwargs) -> Tuple[Image.Image, Image.Image, dict]:
    """
    Align two PIL images and associated masks using an associated method from
    ANTs. Also intakes a binary image mask from PIL to align. Note that all
    images and masks must be of the same size. Uses ANTs as backend.

    Parameters
    ----------
    mov_pil : PIL.Image
        The moving image to be deformed during alignment.
    fix_pil : PIL.Image
        The fixed image to serve as a reference (remain constant) during alignment.
    defaultvalue : int
        If the image is aligned such that the the new image contains points from outside
        the domain of the original image, this is the value that will be inserted as
        a filler. Default is `0`; if white images, use `255` if RGB.
    **kwargs
        Additional arguments passed to `channeltransform`; see its documentation.
    
    Returns
    -------
    moved_pil : PIL.Image.Image
        The deformed moving image.
    moved_mask_pil : PIL.Image.Image
        The deformed moving image mask.
    affine_tfm_info : dict
        Information supplied by ANTs about the forward transform (used to transform the
        moving to the fixed image) as well as the reverse transform. See
        `[ants.registration](https://antspy.readthedocs.io/en/latest/registration.html)`
        for documentation on this output.
    """
    
    assert np.all([isinstance(pil, Image.Image) for pil in [mov_pil, fix_pil, mov_mask_pil]]), \
        'All images must be of type PIL.Image.Image.'
    assert np.all([np.array_equal(pil.size, mov_pil.size) for pil in [fix_pil, mov_mask_pil]]), \
        'All images must be of equal size.'
    
    # Perform the registration. If the images are both empty, this will emit a warning
    # since it uses undefined transforms.
    fixedants=pil2ants(fix_pil.convert('L'))
    moveants=pil2ants(mov_pil.convert('L'))
    
    init_tfm_info = ants.affine_initializer(fixedants, moveants,
                                            use_principal_axis=use_principal_axis,
                                            search_factor=search_factor,  #may be relaxed if major axis is available.
                                            radian_fraction=1,
                                            local_search_iterations=10)
    moveants_ini = ants.apply_transforms(fixed=fixedants,
                                        moving= moveants,
                                        interpolator='linear', #nearestNeighbor
                                        transformlist=init_tfm_info)
    
    # Perform the registration. If the images are both empty, this will emit a warning
    # since it uses undefined transforms.
    tfm_info = ants.registration(fixed=fixedants,
                                 moving=moveants_ini,
                                 type_of_transform=type_of_transform)

    # channeltransform has been updated to handle empty transform lists (just return a copy
    # of the input image).
    mov_aff, mov_mask_aff = map(lambda img, dv, dtype: Image.fromarray(
                                                       channeltransform(img_ants=pil2ants(img),
                                                                        transform=[init_tfm_info] + tfm_info['fwdtransforms'],
                                                                        defaultvalue=dv,
                                                                        **kwargs).numpy().astype(dtype)),
                                (mov_pil, mov_mask_pil),
                                (float(defaultvalue), 0),
                                (np.uint8, bool))

    return mov_aff, mov_mask_aff, tfm_info

def dense_align_pil(mov_pil: Image.Image,
                    fix_pil: Image.Image,
                    mov_mask_pil: Image.Image,
                    fix_mask_pil: Image.Image,
                    model: EnrichedFunctionalModel,
                    patch_shape: Tuple[int, int] = None, # in height/width now for model
                    overlap: int = 0,
                    batch_size: int = 4,
                    origin: str = 'tl',
                    moving_img_name: str = 'moving_img',
                    fixed_img_name: str = 'fixed_img',
                    deff_name: str = 'pos_flow',
                    inv_deff_name: str = 'neg_flow',
                    normalize: bool = True,
                    verbose: bool = False) -> Tuple[Image.Image, np.ndarray, np.ndarray]:
    """
    Align two images `mov_pil` and `fix_pil` and their corresponding masks `mov_mask_pil`
    and `fix_mask_pil` using model `model`.

    The deep learning model `model` should define the properties `dict_inputs` and
    `dict_outputs` that indicate whether the model takes inputs and outputs as dictionaries.
    It should also define the methods `get_input_signature` and `get_output_signature`
    to get the input and output signature in terms of `TensorSpec` objects.

    The model should also output a deformation field and inverse deformation field.

    Example usage:

    ```
    # Make a sample model
    model = SynthMorph(input_shape=(256,256,3),
                       auxiliary_outputs=['def_output', 'inv_def_output'],
                       **simple_efm_kwargs_generator(dict_inputs=True,
                                                     dict_outputs=True,
                                                     outputs_base_name='tensor',
                                                     output_names_list=['moved_img',
                                                                        'def_output',
                                                                        'inv_def_output']))
    print(model.get_input_signature(), model.get_output_signature())

    # Plot the examples.
    example_moving_img = np.sum(np.indices((256,256)), axis=0).astype(np.uint8) * 255
    example_moving_img = np.broadcast_to(np.expand_dims(example_moving_img, -1), (256,256,3))
    example_moving_img = Image.fromarray(example_moving_img)
    example_fixed_img = example_moving_img.copy()
    example_moving_mask = Image.fromarray(np.ones((256,256), dtype=bool))
    example_fixed_mask = Image.fromarray(np.ones((256,256), dtype=bool))
    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.imshow(example_moving_img)
    ax2.imshow(example_fixed_img)
    ```

    Parameters
    ----------
    mov_pil : Image
        Moving image (image to be registered). This should be of PIL format and mode 'RGB'.
    fix_pil : Image
        Fixed image (image to be used as a reference). This should be of PIL format and mode 'RGB'.
    mov_mask_pil : Image
        Moving mask (mask to be registered). This should be of PIL format and mode '1'.
    fix_mask_pil : Image
        Fixed mask (mask to be used as a reference). This should be of PIL format and mode '1'.
    model: EnrichedFunctionalModel
        Deep learning model to be ussed during alignment. This should take in and put out
        dictionary inputs and have the properties dict_inputs, dict_outputs, get_input_signature,
        get_output_signature. See the `EnrichedFunctionalModel` specification for more details.
    patch_shape : tuple[int, int]
        Shape of the patch that the deep learner uses. If left as None (default), this is
        inferred from the input shape of the model. A tuple of the form (H, W) where
        H is the patch height (rows) and W is the patch width (columns).
    overlap : int
        Overlap to use between adjacent patches in pixels.
    batch_size : int
        Batch size to use when aligning patches using the deep learning model `model`.
    origin : str
        The deep learner `model` needs to pad the input image to allow the model to align
        patches correctly. This parameter determines whether the padding is placed on all
        edges of the image ('center', 'c') or the bottom-right corner of the image ('tl',
        'topleft').
    moving_img_name : str
        The name of the moving image as it appears in `model`'s input dictionary.
    fixed_img_name : str
        The name of the fixed image as it appears in `model`'s input dictionary.
    deff_name : str
        The name of the deformation field as it appears in `model`'s output dictionary.
    inv_deff_name : str
        The name of the inverse deformation field as it appears in `model`'s output
        dictionary.
    normalize : bool
        Whether to normalize the input images to the range [0,1] before feeding to
        the deep learner. If the deep learner was already trained on normalized images,
        then this is likely necessary. As PIL images are stored as RGB images, this
        involves simply dividing the image by 255.0.
    verbose : bool
        Whether to print events. Default is False to make this method be quiet.
    """

    assert np.all([isinstance(pil, Image.Image) for pil in [mov_pil,
                                                            fix_pil,
                                                            mov_mask_pil,
                                                            fix_mask_pil]]), \
        'All images must be of type PIL.Image.Image.'
    assert np.all([np.array_equal(pil.size, mov_pil.size) for pil in [fix_pil,
                                                                      mov_mask_pil,
                                                                      fix_mask_pil]]), \
        'All images must be of the same size.'
    
    # Assert the model exists and is an EnrichedFunctionalModel that takes dict inputs
    assert isinstance(model, EnrichedFunctionalModel), \
        'Model must be a subclass of EnrichedFunctionalModel.'
    assert model.dict_inputs, 'The supplied EnrichedFunctionalModel must support dictionary inputs.'
    assert model.dict_outputs, 'The supplied EnrichedFunctionalModel must support dictionary outputs.'
    
    # If patch_shape is None, infer it from the model
    if patch_shape is None:
        first_shape = model._get_input_signature(dict_mode=False)[0].shape
        assert len(first_shape) >= 3, 'Model must take two-dimensional inputs (takes in %d-D).' \
            % (len(first_shape) - 1)
        patch_shape = tuple(first_shape[1:3])

    # Make sure the patch shape is valid and convert it to the patch size.
    assert len(patch_shape) == 2 and isinstance(patch_shape, tuple), 'patch_shape must be a length-2 tuple.'
    #patch_size = tuple(patch_shape[::-1]) # ij to xy indexing

    # Handle overlap
    assert isinstance(overlap, int) and overlap >= 0, 'Overlap must be a non-negative integer.'
    
    ## PROCESSING

    # Convert all images to numpy and pad
    mov, fix, mov_mask, fix_mask = map(lambda img: pad_img_to_even_tile(img=np.array(img).astype(float),
                                                                        patch_shape=patch_shape,
                                                                        overlap=overlap,
                                                                        origin=origin),
                                       (mov_pil, fix_pil, mov_mask_pil, fix_mask_pil))
    assert mov.ndim == 3, 'mov_pil must have a channel dimension (cannot be mode \'1\', \'L\' or related).'
    assert fix.ndim == 3, 'fix_pil must have a channel dimension (cannot be mode \'1\', \'L\' or related).'

    # If normalization is enabled, clamp from 0->255 (PIL) to 0->1.
    if normalize: mov, fix = map(lambda img: img / 255.0, (mov, fix))

    # Generate patches for fine alignment.
    # note that mov and fix are the same shape
    patch_shape_channels = patch_shape + (mov.shape[-1],)
    if np.array_equal(mov.shape, patch_shape_channels):
        # If one patch, just pad with the M and N dimensions
        slices_fix, slices_mov = map(lambda img: img.reshape(1, 1, *img.shape), (fix, mov))
    else:
        # Generate M x N patch grid and enforce overlap
        slices_fix, slices_mov = map(lambda img: np.squeeze(
            np.lib.stride_tricks.sliding_window_view(img,
                                                     patch_shape_channels,
                                                     axis=None))[::patch_shape[0] - overlap,
                                                                 ::patch_shape[1] - overlap],
                                        (fix, mov))

    # Determine valid patches based on a threshold and then make patch generator.
    patch_locs = genpatchlocs(img_shape_2d=mov.shape[:2],
                              patch_shape_2d=patch_shape,
                              overlap=overlap)
    fixed_valid, moving_valid = map(istissue,
                                    (fix_mask, mov_mask),
                                    (patch_locs,) * 2)
    all_valid = (fixed_valid | moving_valid)

    # Predict and correct the patches.
    # ISSUE: Nans will break this
    # ISSUE: Predict will not work with call_seeded
    outputs_dict = model.predict(x={moving_img_name: slices_mov[all_valid],
                                    fixed_img_name: slices_fix[all_valid]},
                                 batch_size=batch_size,
                                 verbose=0)
    
    # Re-add the patches that were excluded from alignment and set them to zero.
    deff_out, inv_deff_out = outputs_dict[deff_name], outputs_dict[inv_deff_name]
    deff_patch_shape = tuple(slices_mov.shape[:-1]) + (2,)
    deff_patch = np.zeros(deff_patch_shape)
    inv_deff_patch = np.zeros(deff_patch_shape)
    deff_patch[all_valid] = deff_out
    inv_deff_patch[all_valid] = inv_deff_out

    # Stitch the deformation field together from the patches.
    global_deff, inv_global_deff = map(lambda patches_grid: stitch(
            patches=patches_grid.reshape(-1, *patches_grid.shape[2:]),
            tiled_shape=patches_grid.shape[:2],
            overlap=overlap,
            multichannel=True,
            window_type='triang',
            order='r',
            suppress_corners=True,
            verbose=verbose),
        (deff_patch, inv_deff_patch))

    # Crop the deformation fields to work with big images.
    mov = unpad_img_from_even_tile(img=mov, orig_shape=mov_pil.size[::-1], origin=origin)
    global_deff, global_inv_deff = map(lambda img: unpad_img_from_even_tile(
            img=img,
            orig_shape=mov.shape[:2], # should be original reg shape
            origin=origin),
        (global_deff, inv_global_deff))

    # Make the moved image.
    movd = vxm.layers.SpatialTransformer(interp_method='linear',
                                         indexing='ij')([tf.expand_dims(mov, axis=0),
                                                         tf.expand_dims(global_deff, axis=0)])[0].numpy()

    # If normalized, un-normalize here. Since this is converted back to uint8 at next step,
    # no need to cast
    if normalize: movd = (movd * 255)
    
    return (Image.fromarray(movd.astype(np.uint8)),
            global_deff,
            global_inv_deff)

def register(im_mov: Image.Image,
             im_fix: Image.Image,
             max_reg_dim: int = 256,
             spatial_mov: np.ndarray = None,
             spatial_strategy: str = 'points',
             raster_point_diameter: float = None,
             mask_mov: Image.Image = None,
             mask_fix: Image.Image = None,        
             mask_fn: Callable[[Image.Image], Image.Image] = None,
             defaultvalue: float = 255,
             apply_masks: bool = True,
             mode: str = 'dense',
             model: EnrichedFunctionalModel = None,
             dense_patch_shape: Tuple[int, int] = None,
             dense_patch_overlap: int = 128,
             dense_batch_size: int = 4,
             dense_origin: str = 'tl',
             dense_moving_img_name: str = 'moving_img',
             dense_fixed_img_name: str = 'fixed_img',
             dense_deff_name: str = 'pos_flow',
             dense_inv_deff_name: str = 'neg_flow',
             dense_normalize: bool = True,
             search_factor=10,
             use_principal_axis=True,
             type_of_transform='Affine',
             verbose: bool = False
             ):
    """
    Completely preprocess, affinely align, and densely align a moving and
    fixed image with optional moving spatial transcriptomics data and masks.
    Images and masks must all be of the same size.

    Parameters
    ----------
    im_mov : PIL.Image.Image
        Moving image that will be aligned during registration.
    im_fix : PIL.Image.Image
        Fixed image that will be used as a refernece (held constant) during
        registration.
    max_reg_dim : int
        When registering, images are downsampled to avoid excessive computational
        complexity. This controls the maximum dimension of the downsampled image
        in pixels. Default is 256.
    spatial_mov : np.ndarray
        An optional array containing ST data of the shape `[n_points, 2]`. You
        can source these directly from scanpy, but make sure the scale of the points
        is the same as the scale of the image being registered. Default is None.
    spatial_strategy : PIL.Image.Image
        Strategy to use when DENSELY aligining spatial_mov data:

        * `'points'` uses the inverse transform as an approximation of how much
          to move ST points. This method is not perfect but should increase
          computational speed. **Currently untested.**
        * `'raster'` converts the points to a raster image that is then deformed
          in the same way that the original image is. The centroid of each point
          is then reread and stored as moved ST point data. A raster image is
          also saved. **For large images,** this is likely to be more correct
          but take more computational power.
        
        Default is 'points'.
    raster_point_diameter : float
        If using spatial_strategy `'raster'`, determines the size of the drawn
        points on the raster image.
    mask_mov : PIL.Image.Image
        Mask for the moving image that will be used to preprocess the moving image
        during registration. If unsupplied, a mask will be generated using
        `mask_fn`. Default is None.
    mask_fix : PIL.Image.Image
        Mask for the fixed image that will be used to preprocess the fixed image
        during registration. If unsupplied, a mask will be generated using
        `mask_fn`. Default is None.
    mask_fn : function
        A function that takes in a single argument (a PIL.Image.Image) and converts
        it to a PIL.Image.Image mask of mode '1'.
    defaultvalue : PIL.Image.Image
        If the image is aligned such that the the new image contains points from outside
        the domain of the original image, this is the value that will be inserted as
        a filler. Default is 255. THIS IS ONLY USED BY DOWNSAMPLED AFFINE IMAGE ALIGNMENT;
        THE ALIGNMENT OF THE FULL IMAGE USES NEAREST-NEIGHBOR EXTRAPOLATION.
    apply_masks : PIL.Image.Image
        Whether to apply masks during registration. Set to False if the images are 
        already masked and you are hoping to save computation time. Default is True.
    mode : str
        Whether to only preprocess the images ('preprocess'), preprocess and affinely align
        the images ('affine'), or to preprocess, affinely align, and then densely align
        the images ('dense'). Default is 'dense'.
    model: EnrichedFunctionalModel
        Deep learning model to be ussed during alignment. This should take in and put out
        dictionary inputs and have the properties dict_inputs, dict_outputs, get_input_signature,
        get_output_signature. See the `EnrichedFunctionalModel` specification for more details.
    dense_patch_shape : tuple[int, int]
        Shape of the patch that the deep learner uses. If left as None (default), this is
        inferred from the input shape of the model. A tuple of the form (H, W) where
        H is the patch height (rows) and W is the patch width (columns).
    dense_patch_overlap : int
        Overlap to use between adjacent patches in pixels.
        batch_size : int
        Batch size to use when aligning patches using the deep learning model `model`.
    dense_batch_size : int
        Batch size to use when aligning patches using the deep learning model `model`.
    dense_origin : str
        The deep learner `model` needs to pad the input image to allow the model to align
        patches correctly. This parameter determines whether the padding is placed on all
        edges of the image ('center', 'c') or the bottom-right corner of the image ('tl',
        'topleft').
    dense_moving_img_name : str
        The name of the moving image as it appears in `model`'s input dictionary.
    dense_fixed_img_name : str
        The name of the fixed image as it appears in `model`'s input dictionary.
    dense_deff_name : str
        The name of the deformation field as it appears in `model`'s output dictionary.
    dense_inv_deff_name : str
        The name of the inverse deformation field as it appears in `model`'s output
        dictionary.
    dense_normalize : bool
        Whether to normalize the input images to the range [0,1] before feeding to
        the deep learner. If the deep learner was already trained on normalized images,
        then this is likely necessary. As PIL images are stored as RGB images, this
        involves simply dividing the image by 255.0.
    search_factor: int
        parameter for the affine initializer. Sufficient search_factor (e.g. 20) is needed for 
        the initialization when large rotations are expected during alignment but it is
        more time consuming. The initializer is off if search_factor < 10 to avoid producing 
        unstable results
    use_principal_axis: bool
        whether to use automatically detected principal axis of the image to perform the initialization
        of the affine alignment. True for faster speed, False for more accurate initialization.
    verbose : bool
        Whether to print events. Default is False to make this method be quiet.   
    

    Returns
    ------
    data_out : dict
        A dictionary containing important registration outputs of the following format:

        ```
        {
            'preprocess': {
                'image': ...,
                'mask': ...,
                'spatial': ...
            },
            'affine': {
                'image': ...,
                'mask': ...,
                'spatial': ...,
                'spatial-raster': ...
            },
            'dense': {
                'image': ...,
                'mask': ...,
                'spatial': ...,
                'spatial-raster': ...
            },
        }
        ```

        `spatial` is only included if `spatial_mov` is not None. `spatial_raster`
        is only included if `spatial_mov` is not None and `spatial_strategy` is
        `'raster'`.
    """

    # Verify the alignment mode
    mode = mode.lower()
    assert mode in ('preprocess', 'affine', 'dense'), \
        "mode must be one of ('preprocess', 'affine', 'dense')."
    assert not (mode == 'dense' and model is None), 'model cannot be None if mode is \'dense\'.'
    
    # Verify the spatial strategy
    spatial_strategy = spatial_strategy.lower()
    assert spatial_strategy in ('points', 'raster'), \
        "spatial_strategy must be one of ('points', 'raster')."
    if spatial_strategy == 'raster':
        assert raster_point_diameter is not None, "raster_point_diameter cannot be None when spatial_strategy is 'raster'."

    # Validate the mask function
    # max reg dim should match the registration-size images later.
    if mask_fn is None: mask_fn = lambda pil_im: standard_mask_fn(pil_im=pil_im,
                                                                  max_mask_dim=max_reg_dim)
    
    # Preprocess and verify the images.
    if not isinstance(im_mov, Image.Image): im_mov = Image.fromarray(im_mov)
    if not isinstance(im_mov, Image.Image): im_fix = Image.fromarray(im_fix)
    assert np.array_equal(im_mov.size, im_fix.size), 'Anndata objects must reference images of the same size.'
    
    # Preprocess and verify the masks if a masking function present.
    if mask_mov is not None:
        if not isinstance(mask_mov, Image.Image): mask_mov = Image.fromarray(mask_mov)
    elif mask_fn is not None: mask_mov = mask_fn(im_mov)
    assert np.array_equal(im_mov.size, mask_mov.size) and mask_mov.mode == '1', \
            'mask_mov must be a binary PIL mask with the same size as im_mov.'
    if mask_fix is not None:
        if not isinstance(mask_fix, Image.Image): mask_fix = Image.fromarray(mask_fix)
    elif mask_fn is not None: mask_fix = mask_fn(im_fix)
    assert np.array_equal(im_fix.size, mask_fix.size) and mask_fix.mode == '1', \
            'mask_fix must be a binary PIL mask with the same size as im_fix.'
    
    # verify the spatial coordinates.
    if spatial_mov is not None: assert spatial_mov.ndim == 2 and spatial_mov.shape[-1] == 2, \
        'spatial_mov must be either None or an array-like structure with shape (N, 2).'
    
    # APPLY MASKS TO INITIAL IMAGES

    if apply_masks:
        im_mov, im_fix = map(np.array, (im_mov, im_fix))
        im_mov[~np.asarray(mask_mov).astype(bool)] = defaultvalue
        im_fix[~np.asarray(mask_fix).astype(bool)] = defaultvalue
        im_mov, im_fix = map(Image.fromarray, (im_mov, im_fix))

    # Update output data
    data_out = {'preprocess': {'moving-image': im_mov,
                               'fixed-image': im_fix,
                               'moving-mask': mask_mov,
                               'fixed-mask': mask_fix}}
    if spatial_mov is not None:
        data_out['preprocess']['spatial'] = spatial_mov

    

    if mode in ('affine', 'dense'):
        # AFFINE REGISTRATION
        
        # downsample all structures to the maximum registration dimension.
        # no need to do spatial since that will be done on full images
        im_mov_r, im_fix_r, mask_mov_r, mask_fix_r = map(lambda x, resample: resize_to_max_dim_pil(x,
                                                                                                max_dim=max_reg_dim,
                                                                                                resample=resample),
                                                        (im_mov, im_fix, mask_mov, mask_fix),
                                                        (Image.Resampling.BILINEAR,
                                                        Image.Resampling.BILINEAR,
                                                        Image.Resampling.NEAREST,
                                                        Image.Resampling.NEAREST))
        
        # perform affine registration; omit itk for now.
        if search_factor>=20:
         #print('initializer is on')
         im_mov_ra, mask_mov_ra, aff_tfm_info = ants_align_pil2(mov_pil=im_mov_r,
                                                              fix_pil=im_fix_r,
                                                              mov_mask_pil=mask_mov_r,
                                                              defaultvalue=defaultvalue,
                                                              type_of_transform=type_of_transform,
                                                              search_factor=search_factor,
                                                              use_principal_axis=use_principal_axis)
        else:
         im_mov_ra, mask_mov_ra, aff_tfm_info = ants_align_pil(mov_pil=im_mov_r,
                                                                fix_pil=im_fix_r,
                                                                mov_mask_pil=mask_mov_r,
                                                                defaultvalue=defaultvalue,
                                                                type_of_transform=type_of_transform)

        # if there is indication for a transform at all, do it
        # else, just copy the original structures. Make note that at this point the mask
        # becomes RGB for ease of use
        if len(aff_tfm_info['fwdtransforms']) > 0:
            # upscale the affine transform
            large_aff_tfm = ants.read_transform(aff_tfm_info['fwdtransforms'][0]) # assume only one matrix out.
            large_aff_tfm = scale_affine(trf=large_aff_tfm,
                                        orig_shape_2d=im_mov_r.size[::-1],
                                        new_shape_2d=im_mov.size[::-1])
            #@ check if large_aff_tfm misses rotation or ants2itk_affine misses
            large_aff_tfm_itk = ants2itk_affine(large_aff_tfm)

            # align the large images
            im_mov_a, mask_mov_a = map(lambda input, interp, dv: apply_itk_trf_image(
                input=input,
                trf=large_aff_tfm_itk,
                interpolator=interp,
                defaultPixelValue=dv,
                outputPixelType=sitk.sitkUnknown,
                useNearestNeighborExtrapolator=True),
                                    (np.asarray(im_mov).astype(float),
                                        np.asarray(mask_mov.convert('RGB')).astype(float)),
                                    (sitk.sitkLinear, sitk.sitkNearestNeighbor),
                                    (float(defaultvalue), 0.0))
        else:
            im_mov_a = np.array(im_mov).astype(float)
            mask_mov_a = np.array(mask_mov.convert('RGB')).astype(float)

        # update the output data
        data_out['affine'] = {'image': Image.fromarray(im_mov_a.astype(np.uint8)),
                              'mask': Image.fromarray(np.mean(mask_mov_a, axis=-1).astype(bool))}
        inv_large_aff_tfm_itk=large_aff_tfm_itk.GetInverse()
        if spatial_mov is not None:
            if spatial_strategy == 'points':
                # need to apply inverse transform to mimic image's resample.
                spatial_mov_a = register_spatial_with_itk_points(
                    spatial_data=spatial_mov,
                    inverse_itk_trf=large_aff_tfm_itk.GetInverse(), 
                    spatial_data_indexing='xy')
            elif spatial_strategy == 'raster':
                spatial_mov_a, spatial_raster = register_spatial_with_itk_raster(
                    spatial_data=spatial_mov,
                    inverse_itk_trf=large_aff_tfm_itk.GetInverse(),
                    spatial_data_indexing='xy',
                    spatial_raster_size=im_mov.size,
                    raster_point_diameter=raster_point_diameter)
                data_out['affine']['spatial-raster'] = spatial_raster
            data_out['affine']['spatial'] = spatial_mov_a
            data_out['affine']['inv_deff'] = inv_large_aff_tfm_itk
            data_out['affine']['deff'] = large_aff_tfm_itk
            
    if mode == 'dense':
        # DENSE REGISTRATION

        # densely align the small images
        _, global_deff, global_inv_deff = dense_align_pil(mov_pil=im_mov_ra,
                                                          fix_pil=im_fix_r,
                                                          model=model,
                                                          mov_mask_pil=mask_mov_ra,
                                                          fix_mask_pil=mask_fix_r,
                                                          patch_shape=dense_patch_shape, # in width/height since pil
                                                          overlap=dense_patch_overlap,
                                                          batch_size=dense_batch_size,
                                                          origin=dense_origin,
                                                          moving_img_name=dense_moving_img_name,
                                                          fixed_img_name=dense_fixed_img_name,
                                                          deff_name=dense_deff_name,
                                                          inv_deff_name=dense_inv_deff_name,
                                                          normalize=dense_normalize,
                                                          verbose=verbose)
        
        # transition from small to large transform
        large_global_deff, large_global_inv_deff = map(lambda deff: coords2patch(struct=deff,
                                                                                    coords=(0,0) + tuple(deff.shape[:2]), #ULeLoR
                                                                                    backend='ne',
                                                                                    out_patch_shape=im_mov.size[::-1],
                                                                                    interp_method='linear',
                                                                                    fill_value=None),
                                                    (global_deff, global_inv_deff))
        
        # align the large images
        im_mov_d, mask_mov_d = map(
            lambda arr: vxm.layers.SpatialTransformer(interp_method='linear',
                                                    indexing='ij')([tf.expand_dims(arr.astype(float), axis=0),
                                                                    tf.expand_dims(large_global_deff.astype(float), axis=0)])[0].numpy(),
            (im_mov_a, mask_mov_a))
        im_mov_d = Image.fromarray(im_mov_d.astype(np.uint8))
        mask_mov_d = Image.fromarray(np.mean(mask_mov_d, axis=-1).astype(bool)).convert('1')
        if apply_masks:
            im_mov_d = np.array(im_mov_d)
            im_mov_d[~np.asarray(mask_mov_d).astype(bool)] = defaultvalue
            im_mov_d = Image.fromarray(im_mov_d)
        
        # Update the output data
        data_out['dense'] = {'image': im_mov_d,
                             'mask': mask_mov_d}
        if spatial_mov is not None:
            if spatial_strategy == 'points':
                spatial_mov_d = register_spatial_with_def_field_points(
                    spatial_data=spatial_mov_a,
                    inverse_def_field=large_global_inv_deff,
                    spatial_data_indexing='xy',
                    inverse_def_field_indexing='ij')
            data_out['dense']['spatial'] = spatial_mov_d
            data_out['dense']['deff']=large_global_deff
            data_out['dense']['inv_deff']=large_global_inv_deff
    
    return data_out

# might be worthwhile to consider adding old (unregistered) info to archive under .uns
def sc_register(adata_mov,
                adata_fix,
                max_reg_dim: int = 256,
                spatial_target: str = 'hires',
                spatial_strategy: str = 'points',
                mask_fn=standard_mask_fn,
                apply_masks=True,
                defaultvalue=0, #background color is 0
                model=None,
                dense_patch_shape=None,
                dense_patch_overlap=128,
                dense_batch_size: int = 4,
                dense_origin: str = 'tl',
                dense_moving_img_name='moving_img',
                dense_fixed_img_name='fixed_img',
                dense_deff_name='pos_flow',
                dense_inv_deff_name='neg_flow',
                dense_normalize=True,
                inplace=False,
                spot_diameter_unscaled=None,
                mode='dense',
                search_factor=20,
                use_principal_axis=False,
                type_of_transform='Affine',
                verbose=False):
    """
    Completely preprocess, affinely align, and densely align a moving and
    fixed image with optional moving spatial transcriptomics data and masks.
    Images and masks must all be of the same size.

    Parameters
    ----------
    adata_mov : AnnData
        AnnData object with the moving image that will be aligned during registration.
    adata_fix : AnnData
        AnnData object with the fixed image that will be used as a refernece (held constant)
        during registration.
    max_reg_dim : int
        When registering, images are downsampled to avoid excessive computational
        complexity. This controls the maximum dimension of the downsampled image
        in pixels. Default is 256.
    spatial_target : str
        The category of spatial target to use in the AnnData files. The default is
        'hires' since that is usually the location at which AnnData image objects are stored.
        See Anndata.uns for more details.
    spatial_strategy : PIL.Image.Image
        Strategy to use when DENSELY aligining spatial_mov data:

        * `'points'` uses the inverse transform as an approximation of how much
          to move ST points. This method is not perfect but should increase
          computational speed. **Currently untested.**
        * `'raster'` converts the points to a raster image that is then deformed
          in the same way that the original image is. The centroid of each point
          is then reread and stored as moved ST point data. A raster image is
          also saved. **For large images,** this is likely to be more correct
          but take more computational power.
        
        Default is 'points'.
    raster_point_diameter : float
        If using spatial_strategy `'raster'`, determines the size of the drawn
        points on the raster image.
    mask_fn : function
        A function that takes in a single argument (a PIL.Image.Image) and converts
        it to a PIL.Image.Image mask of mode '1'.
        Default 'standard_mask_fn' only works for RGB images but not grayscale images
    defaultvalue : PIL.Image.Image
        If the image is aligned such that the the new image contains points from outside
        the domain of the original image, this is the value that will be inserted as
        a filler. Default is 255. THIS IS ONLY USED BY DOWNSAMPLED AFFINE IMAGE ALIGNMENT;
        THE ALIGNMENT OF THE FULL IMAGE USES NEAREST-NEIGHBOR EXTRAPOLATION.
    apply_masks : PIL.Image.Image
        Whether to apply masks during registration. Set to False if the images are 
        already masked and you are hoping to save computation time. Default is True.
    mode : str
        Whether to only preprocess the images ('preprocess'), preprocess and affinely align
        the images ('affine'), or to preprocess, affinely align, and then densely align
        the images ('dense'). Default is 'dense'.
    model: EnrichedFunctionalModel
        Deep learning model to be ussed during alignment. This should take in and put out
        dictionary inputs and have the properties dict_inputs, dict_outputs, get_input_signature,
        get_output_signature. See the `EnrichedFunctionalModel` specification for more details.
    dense_patch_shape : tuple[int, int]
        Shape of the patch that the deep learner uses. If left as None (default), this is
        inferred from the input shape of the model. A tuple of the form (H, W) where
        H is the patch height (rows) and W is the patch width (columns).
    dense_patch_overlap : int
        Overlap to use between adjacent patches in pixels.
        batch_size : int
        Batch size to use when aligning patches using the deep learning model `model`.
    dense_batch_size : int
        Batch size to use when aligning patches using the deep learning model `model`.
    dense_origin : str
        The deep learner `model` needs to pad the input image to allow the model to align
        patches correctly. This parameter determines whether the padding is placed on all
        edges of the image ('center', 'c') or the bottom-right corner of the image ('tl',
        'topleft').
    dense_moving_img_name : str
        The name of the moving image as it appears in `model`'s input dictionary.
    dense_fixed_img_name : str
        The name of the fixed image as it appears in `model`'s input dictionary.
    dense_deff_name : str
        The name of the deformation field as it appears in `model`'s output dictionary.
    dense_inv_deff_name : str
        The name of the inverse deformation field as it appears in `model`'s output
        dictionary.
    dense_normalize : bool
        Whether to normalize the input images to the range [0,1] before feeding to
        the deep learner. If the deep learner was already trained on normalized images,
        then this is likely necessary. As PIL images are stored as RGB images, this
        involves simply dividing the image by 255.0.
    search_factor: int
        parameter for the affine initializer. Sufficient search_factor (e.g. 20) is needed for 
        the initialization when large rotations are expected during alignment but it is
        more time consuming. The initializer is off if search_factor < 10 to avoid producing 
        unstable results
    use_principal_axis: bool
        whether to use automatically detected principal axis of the image to perform the initialization
        of the affine alignment. True for faster speed, False for more accurate initialization.
    verbose : bool
        Whether to print events. Default is False to make this method be quiet.    

    Returns
    ------
    data_out : tuple[dict, AnnData]
        A tuple where the first element contains a dictionary with important registration outputs
        of the following format:

        ```
        {
            'preprocess': {
                'image': ...,
                'mask': ...,
                'spatial': ...
            },
            'affine': {
                'image': ...,
                'mask': ...,
                'spatial': ...,
                'spatial-raster': ...
            },
            'dense': {
                'image': ...,
                'mask': ...,
                'spatial': ...,
                'spatial-raster': ...
            },
        }
        ```

        and the second element is the output AnnData object.

        For the first dictionary, note that `spatial` is only included if `spatial_mov` 
        is not None. `spatial_raster` is only included if `spatial_mov` is not None and
        `spatial_strategy` is `'raster'`.
    """
   
    
    # Each anndata should only have one spatial container
    assert len(adata_mov.uns['spatial']) == 1 and len(adata_fix.uns['spatial']) == 1, \
        'Both anndata objects may only have one entry in .uns[\'spatial\'].'
    spatial_mov, spatial_fix = map(lambda adata: next(iter(adata.uns['spatial'].values())),
                                   (adata_mov, adata_fix))
    # PATCH START 
    # STaCker assumes 'hires' by default, but we use 'density'
    if spatial_target is None or spatial_target not in spatial_mov['images']:
        if 'density' in spatial_mov['images']:
            spatial_target = 'density'
        else:
            raise KeyError(
                f"Invalid spatial_target. Got {spatial_target}. "
                f"Available image keys: {list(spatial_mov['images'].keys())}"
            )

    
    # Extract the moving and fixed images and associated data
    im_mov, im_fix = map(lambda scont: Image.fromarray((scont['images'][spatial_target] * 255)
                                                                   .astype(np.uint8)).convert('RGB'),
                         (spatial_mov, spatial_fix))
    scalefactor_mov = spatial_mov['scalefactors']['tissue_%s_scalef' % spatial_target]
    spatial_mov_scaled = adata_mov.obsm['spatial'] * scalefactor_mov
    if spatial_strategy.lower() == 'raster':
        if spot_diameter_unscaled is None: spot_diameter_unscaled = spatial_mov['scalefactors']['spot_diameter_fullres']
        spot_diameter_scaled = spot_diameter_unscaled * scalefactor_mov
    else: spot_diameter_scaled = None
    
    # Assert the data are valid.
    assert np.array_equal(im_mov.size, im_fix.size), 'Anndata objects must reference images of the same size.'
    
    # Register the images using another function.
    rout = register(im_mov,
                    im_fix,
                    max_reg_dim=max_reg_dim,
                    spatial_mov=spatial_mov_scaled,
                    spatial_strategy=spatial_strategy,
                    raster_point_diameter=spot_diameter_scaled,
                    mask_mov=spatial_mov['images'].get(spatial_target + '_mask', None),
                    mask_fix=spatial_fix['images'].get(spatial_target + '_mask', None),
                    mask_fn=mask_fn,
                    apply_masks=apply_masks,
                    defaultvalue=defaultvalue,
                    dense_patch_shape=dense_patch_shape,
                    dense_patch_overlap=dense_patch_overlap,
                    dense_moving_img_name=dense_moving_img_name,
                    dense_fixed_img_name=dense_fixed_img_name,
                    dense_deff_name=dense_deff_name,
                    dense_inv_deff_name=dense_inv_deff_name,
                    dense_batch_size=dense_batch_size,
                    dense_origin=dense_origin,
                    model=model,
                    mode=mode,
                    dense_normalize=dense_normalize,
                    search_factor=search_factor,
                    use_principal_axis=use_principal_axis,
                    type_of_transform=type_of_transform,
                    verbose=verbose)
    
    # Update information in the destination anndata.
    adata_movd = adata_mov if inplace else adata_mov.copy()
    movd_spatial_nm = next(iter(adata_movd.uns['spatial'].keys()))
    adata_movd.uns['spatial'][movd_spatial_nm]['images'][spatial_target] = np.asarray(rout[mode]['image']) / 255
    adata_movd.obsm['spatial'] = rout[mode]['spatial'] / scalefactor_mov
                                
    return adata_movd, rout


def stacker_register(slices,
                alignment_mode,
                ref_index: int = 0,
                max_reg_dim: int = 512,
                spatial_target: str = 'hires',
                spatial_strategy: str = 'points',
                mask_fn=standard_mask_fn,
                apply_masks=True,
                defaultvalue=0, #background color is 0
                model=None,
                dense_patch_shape=None,
                dense_patch_overlap=128,
                dense_batch_size: int = 4,
                dense_origin: str = 'tl',
                dense_moving_img_name='moving_img',
                dense_fixed_img_name='fixed_img',
                dense_deff_name='pos_flow',
                dense_inv_deff_name='neg_flow',
                dense_normalize=True,
                inplace=False,
                spot_diameter_unscaled=None,
                mode='dense',
                search_factor=5,
                use_principal_axis=False,
                type_of_prep='Affine',  
                output_dff=False,
                verbose=False):
    """
    Completely preprocess, affinely align, and densely align a moving and
    fixed image with optional moving spatial transcriptomics data and masks.
    Images and masks must all be of the same size.

    Parameters
    ----------
    slices : list of AnnData
        list of AnnData objects to be aligned.
    alignment_mode : str
        'templated' or 'templateless'; if templated, ref_index must be defined
    ref_index: int
        the index of the reference slice in the input list of slices (default: 0); 
        it refers to the fixed slice when the alignment_mode is 'templated';
    max_reg_dim : int
        When registering, images are downsampled to avoid excessive computational
        complexity. This controls the maximum dimension of the downsampled image
        in pixels. Default is 256.
    spatial_target : str
        The category of spatial target to use in the AnnData files. The default is
        'hires' since that is usually the location at which AnnData image objects are stored.
        See Anndata.uns for more details.
    spatial_strategy : PIL.Image.Image
        Strategy to use when DENSELY aligining spatial_mov data:
        * `'points'` uses the inverse transform as an approximation of how much
          to move ST points. This method is not perfect but should increase
          computational speed. 
    mask_fn : function
        A function that takes in a single argument (a PIL.Image.Image) and converts
        it to a PIL.Image.Image mask of mode '1'.
    defaultvalue : PIL.Image.Image
        If the image is aligned such that the the new image contains points from outside
        the domain of the original image, this is the value that will be inserted as
        a filler. Default is 255. THIS IS ONLY USED BY DOWNSAMPLED AFFINE IMAGE ALIGNMENT;
        THE ALIGNMENT OF THE FULL IMAGE USES NEAREST-NEIGHBOR EXTRAPOLATION.
    apply_masks : PIL.Image.Image
        Whether to apply masks during registration. Set to False if the images are 
        already masked and you are hoping to save computation time. Default is True.
    mode : str
        Whether to only preprocess the images ('preprocess'), preprocess and affinely align
        the images ('affine'), or to preprocess, affinely align, and then densely align
        the images ('dense'). Default is 'dense'.
    model: EnrichedFunctionalModel
        Deep learning model to be ussed during alignment. This should take in and put out
        dictionary inputs and have the properties dict_inputs, dict_outputs, get_input_signature,
        get_output_signature. See the `EnrichedFunctionalModel` specification for more details.
    dense_patch_shape : tuple[int, int]
        Shape of the patch that the deep learner uses. If left as None (default), this is
        inferred from the input shape of the model. A tuple of the form (H, W) where
        H is the patch height (rows) and W is the patch width (columns).
    dense_patch_overlap : int
        Overlap to use between adjacent patches in pixels.
        batch_size : int
        Batch size to use when aligning patches using the deep learning model `model`.
    dense_batch_size : int
        Batch size to use when aligning patches using the deep learning model `model`.
    dense_origin : str
        The deep learner `model` needs to pad the input image to allow the model to align
        patches correctly. This parameter determines whether the padding is placed on all
        edges of the image ('center', 'c') or the bottom-right corner of the image ('tl',
        'topleft').
    dense_moving_img_name : str
        The name of the moving image as it appears in `model`'s input dictionary.
    dense_fixed_img_name : str
        The name of the fixed image as it appears in `model`'s input dictionary.
    dense_deff_name : str
        The name of the deformation field as it appears in `model`'s output dictionary.
    dense_inv_deff_name : str
        The name of the inverse deformation field as it appears in `model`'s output
        dictionary.
    dense_normalize : bool
        Whether to normalize the input images to the range [0,1] before feeding to
        the deep learner. If the deep learner was already trained on normalized images,
        then this is likely necessary. As PIL images are stored as RGB images, this
        involves simply dividing the image by 255.0.
    search_factor: int
        parameter for the affine initializer. Sufficient search_factor (e.g. 20) is needed for 
        the initialization when large rotations are expected during alignment but it is
        more time consuming. The initializer is off if search_factor < 10 to avoid producing 
        unstable results
    use_principal_axis: bool
        whether to use automatically detected principal axis of the image to perform the initialization
        of the affine alignment. True for faster speed, False for more accurate initialization.
    verbose : bool
        Whether to print events. Default is False to make this method be quiet.    

    Returns
    ------
    data_out : list[AnnData]
        A list of post-alignment spatial transcriptome slices.
        If the alignment_mode is 'templated', the element at the index of 'ref_index' is the fixed slice that
        remains unchanged.
    """
   
    assert len(slices) >= 2 , \
        'At least two spatial slices are needed.' 
    assert alignment_mode == 'templated' or alignment_mode=='templateless' , \
        'Alignment mode must be one of {"templated","templateless"}.' 
    aligned=[None]*len(slices)

    if alignment_mode == 'templated':
        aligned=[None]*len(slices)
        adata_fix=slices[ref_index]
        for each in range(len(slices)):
            if each!=ref_index:
             adata_mov=slices[each]   
             rs=sc_register(adata_mov,
                adata_fix,
                max_reg_dim=max_reg_dim,
                spatial_target=spatial_target,
                spatial_strategy=spatial_strategy,
                mask_fn=mask_fn,
                apply_masks=apply_masks,
                defaultvalue=defaultvalue, 
                model=model,
                dense_patch_shape=dense_patch_shape,
                dense_patch_overlap=dense_patch_overlap,
                dense_batch_size=dense_batch_size,
                dense_origin=dense_origin,
                dense_moving_img_name=dense_moving_img_name,
                dense_fixed_img_name=dense_fixed_img_name,
                dense_deff_name=dense_deff_name,
                dense_inv_deff_name=dense_inv_deff_name,
                dense_normalize=dense_normalize,
                inplace=inplace,
                spot_diameter_unscaled=spot_diameter_unscaled,
                mode=mode,
                search_factor=search_factor,
                use_principal_axis=use_principal_axis,
                type_of_transform=type_of_prep,
                verbose=verbose)
             if output_dff:
                 dff={'af_deff':rs[1]['affine']['deff'],'af_inv_deff':rs[1]['affine']['inv_deff'],
                 'dense_deff':rs[1]['dense']['deff'],'dense_inv_deff':rs[1]['dense']['inv_deff']}
                 aligned[each]=[rs[0],dff]
                 aligned[ref_index]=[slices[ref_index],{}]
             else:
                 aligned[each]=rs[0]
                 aligned[ref_index]=slices[ref_index]
                
    if alignment_mode=='templateless':
        for each in range(len(slices)):
            rest=set(range(len(slices))).difference(set([each]))

            spatial_key=list(slices[each].uns.keys())[0]
            library_key=list(slices[each].uns[spatial_key].keys())[0]

            adata_warped=slices[each].copy()
            final_spatial=adata_warped.obsm['spatial']
            for r in rest:
                adata_unwarped=slices[r].copy()
                rs = sc_register(adata_mov=adata_warped,
                        adata_fix=adata_unwarped,
                        max_reg_dim=max_reg_dim,
                        spatial_target=spatial_target,
                        spatial_strategy=spatial_strategy,
                        mask_fn=mask_fn,
                        apply_masks=apply_masks,
                        defaultvalue=defaultvalue, 
                        model=model,
                        dense_patch_shape=dense_patch_shape,
                        dense_patch_overlap=dense_patch_overlap,
                        dense_batch_size=dense_batch_size,
                        dense_origin=dense_origin,
                        dense_moving_img_name=dense_moving_img_name,
                        dense_fixed_img_name=dense_fixed_img_name,
                        dense_deff_name=dense_deff_name,
                        dense_inv_deff_name=dense_inv_deff_name,
                        dense_normalize=dense_normalize,
                        inplace=inplace,
                        spot_diameter_unscaled=spot_diameter_unscaled,
                        mode=mode,
                        search_factor=search_factor,
                        use_principal_axis=use_principal_axis,
                        type_of_transform=type_of_prep,        
                        verbose=verbose)
                final_spatial=final_spatial+rs[0].obsm['spatial']   
        
            final_spatial=final_spatial/len(slices)
            aligned[each]=slices[each].copy()
            aligned[each].obsm['spatial']=final_spatial
            aligned[each].uns[spatial_key][library_key]['images'][spatial_target]=None 
            
    return aligned        

