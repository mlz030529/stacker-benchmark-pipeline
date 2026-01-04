"""
UTILS.IO

Functions for general data loading and saving.

Author: Peter Lais
Last updated: 10/15/2022
"""

from os import scandir, makedirs
from pathlib import Path
import requests
from matplotlib.pyplot import imsave
import warnings
from typing import Collection, Union
import numpy as np
import tensorflow as tf
from ..utils import normalize
from ..constants import EXAMPLE_DESCRIPTION, EXAMPLE_OUT_TYPES, EXAMPLE_OUT_SHAPES

## INTRODUCED IN V1
# Legacy inclusions that haven't changed
# These may seem unneccesary but are needed if users want to import them.

# Mapping from primitives to features.
def is_scalar(value):
    """
    Determine if a value is scalar.

    Parameters
    ----------
    value : any
        Value to evaluate.
    
    Returns
    -------
    bool
        Whether the value is scalar.
    """

    try:
        len(value)
        return True
    except:
        return False

def byte_feature(value):
    """
    Convert a value to a ByteFeature for writing to disk.

    Parameters
    ----------
    value : any
        Value to convert.
    
    Returns
    -------
    tf.train.Feature
        Output feature.
    """

    if isinstance(value, tf.Tensor):
        value = value.numpy() # Apparently, strings won't be unpacked from EagerTensors
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def int64_feature(value):
    """
    Convert a value to an Int64Feature for writing to disk.

    Parameters
    ----------
    value : any
        Value to convert.
    
    Returns
    -------
    tf.train.Feature
        Output feature.
    """

    if not is_scalar(value): value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def float_feature(value):
    """
    Convert a value to a FloatFeature for writing to disk.

    Parameters
    ----------
    value : any
        Value to convert.
    
    Returns
    -------
    tf.train.Feature
        Output feature.
    """

    if not is_scalar(value): value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def create_example(**kwargs):
    """
    Convert features to an Example for writing to disk.

    Parameters
    ----------
    kwargs
        Name-value pairs of argument names and tf.train.Features to be
        written to disk as Examples.
    
    Returns
    -------
    tf.train.Example
        Output example.
    """

    return tf.train.Example(features=tf.train.Features(feature=kwargs))

def tf_synthetic_record_iterator(filepath, description, out_types):
    """
    A function that reads a TFRecord file and, given a description about the format
    of tf.train.Examples stored in the file, converts each serialized Example
    to its corresponding Tensorflow types for use in regular Python code. Each
    example is then yielded in an iterator-like fashion.

    See https://www.tensorflow.org/tutorials/load_data/tfrecord for more.

    Parameters
    ----------
    filepath : str
        Filepath of the TFRecord file to read.
    description : dict
        A dictionary containing key-value pairs of the form `name: tf.io.Feature (FixedLenFeature, etc.)`.
        This will be useful for parsing the serialized Example information.
    out_types : dict
        A dictionary containing key-value pairs of the form `name: tf.dtype`. These will be used to convert
        arguments into their corresponding Tensorflow types.

    Yields
    ------
    Deserialized example output.
    """
    # Pure iterator for use outside of tf datasets.
    tfrds = tf.data.TFRecordDataset(filepath)
    for serialized_example in tfrds:
        yield serialized_example_to_tensors(serialized_example, description, out_types)

def serialized_example_to_tensors(serialized_example, description, out_types, out_shapes=None):
    """
    Convenience method for converting a serialized example into a dictionary of
    deserialized tensors of the appropriate datatypes and, optionally, output shapes.

    See https://www.tensorflow.org/tutorials/load_data/tfrecord for more.

    Parameters
    ----------
    serialized_example : tf.train.Example
        A serialized example to deserialize.
    description : dict
        A dictionary containing key-value pairs of the form `name: tf.io.Feature (FixedLenFeature, etc.)`.
        This will be useful for parsing the serialized Example information.
    out_types : dict
        A dictionary containing key-value pairs of the form `name: tf.dtype`. These will be used to convert
        arguments into their corresponding Tensorflow types.
    out_shapes : dict
        A dictionary containing key-value pairs of the form `name: tuple`. These will be used to make
        shape information available if tensors are used in Graph mode.
    
    Returns
    -------
    deserialized_tensors : dict
        Deserialized tensors.
    """

    # compatible with mapping a TFRecordDataset to an iterable tensorflow dataset.
    deserialized_example = tf.io.parse_single_example(serialized_example, description)
    deserialized_tensors = {}
    for key, val in deserialized_example.items():
        deserialized_tensors[key] = tf.io.parse_tensor(val, out_type=out_types[key])
    return deserialized_tensors if out_shapes is None else ensure_shapes(deserialized_tensors, out_shapes)

def ensure_shapes(tensor_dict, out_shapes):
    """
    Convenience method for ensuring tensors have defined shapes in graph mode.

    Parameters
    ----------
    tensor_dict : dict
        Dictionary of tensor values whose shapes must be ensured.
    out_shapes : dict
        A dictionary containing key-value pairs of the form `name: tuple`. These will be used to make
        shape information available if tensors are used in Graph mode.
    
    Returns
    -------
    dict
        Tensors with ensured shapes.
    """

    # Ensures shapes are set properly and known to TensorFlow.
    out = {}
    for key, val in tensor_dict.items():
        out[key] = tf.ensure_shape(val, out_shapes[key])
    return out

def remove_nans_from_dict(in_dict, warn=True):
    """
    Remove NaN values from all tensors in a dictionary and replace them with zeros.

    Parameters
    ----------
    in_dict : dict
        Dictionary whose tensors must be screened for NaNs.
    warn : bool
        Whether to emit a warning if NaN values are found. Default is True.
    """
    out = {}
    have_warned = False
    for key, val in in_dict.items():
        out[key] = tf.where(tf.math.is_nan(val), tf.zeros_like(val), val)
        # Issue: figure out if this warning is legit or not.
        if warn and not tf.reduce_all(out[key] == val) and not have_warned:
            warnings.warn('Nans exist in the input tensor. Consider removing them.')
            have_warned=True
    return out

def map_dict_to_v4(in_dict, include_lbl=True):
    """
    Map inputs from a dictionary to a format compatible with the input of
    a SynthMorph (v4) model:

    ```
    [input, output] where

    input = [moving_img, fixed_img, moving_lbl]
    output = fixed_lbl
    ```

    Parameters
    ----------
    in_dict : dict
        The dictionary containing relevant tensor values.
    include_lbl : bool
        Whether to include labels in the returned structure. If not, return 
        a zero tensor for the moving label. Default is True.
    
    Returns
    -------
    tuple
        Tensors arranged in the appropriate format.
    """
    if include_lbl:
        return ((in_dict['moving_img'], in_dict['fixed_img'], in_dict['moving_lbl']), in_dict['fixed_lbl'])
    else:
        # Don't return the moving label
        return ((in_dict['moving_img'], in_dict['fixed_img']), tf.constant(0))

def _sig2tensors(tensor_dict, output_signature):
    """
    NEEDS DOCUMENTATION. Build an output based on an output signature and tensor dictionary.
    """
    # If we have a TensorSpec element, return its corresponding tensor in the dictionary.
    # If the element is not a TensorSpec, we assume it is a list and iterate through that
    # recursively.
    return tuple([tensor_dict[ts.name] if isinstance(ts, tf.TensorSpec)
        else _sig2tensors(tensor_dict, ts) for ts in output_signature])


def dataset_from_slices_with_signature(output_signature, **kwargs):
    """
    NEEDS DOCUMENTATION. Makes a Tensorflow dataset from the slices 
    """
    return tf.data.Dataset.from_tensor_slices(_sig2tensors(tensor_dict=kwargs,
                                                           output_signature=output_signature))


def yield_moving_fixed_slices(slices_mov, slices_fix, patch_shape=None, all_valid=None):
    """
    Yield moving and fixed slices with a given patch shape as a generator for
    Synthmorph (v4) models.

    Parameters
    ----------
    slices_mov : tf.Tensor
        Tensor consisting of a collection of moving slices. Shape should be
        (M, N, H, W, C) where (M, N) forms a 2D collection of patches,
        H is image height, W is image width, and C is image channels.
    slices_fix : tf.Tensor
        Tensor consisting of a collection of fixed slices. Shape should be
        (M, N, H, W, C) where (M, N) forms a 2D collection of patches,
        H is image height, W is image width, and C is image channels.
    all_valid : tf.Tensor or None
        A boolean tensor of shape (M, N) specifying which tensors from slices_mov,
        slices_fix to return. Default is None, which assumes all slices are valid.
    """

    # if no all-valid is supplied, assume all slices are valid.
    if all_valid is not None: all_valid = np.ones(slices_mov.shape[:2], dtype=bool)
    def tfds_gen_fn():
        for moving, fixed in zip(slices_mov[all_valid],
                                 slices_fix[all_valid]):
            moving, fixed = map(lambda x: tf.convert_to_tensor(x / 255.0, dtype=tf.float32),
                                (moving, fixed))
            yield ((moving, fixed), tf.constant(0.0))
    # Generate a tensorflow dataset of the patches.
    tfds = tf.data.Dataset.from_generator(
        tfds_gen_fn,
        output_signature=((tf.TensorSpec(shape=patch_shape, dtype=tf.float32),
                            tf.TensorSpec(shape=patch_shape, dtype=tf.float32)),
                            tf.TensorSpec(shape=(), dtype=tf.float32)))
    return tfds

def slice_and_save(base_fn, ext, tens, axis=-1):
    """
    Slice a 3D structure and save it to a directory under the names
    `base_fn[slice_no][ext]`.

    Parameters
    ----------

    base_fn : str
        The base filename/path that slice numbers will be appended to
        and where slices will be saved.
    ext : str
        An extension of the form `.ext` or `ext` that determines the
        type of file to save as. Currently, only `.png` and `.npy`
        are supported.
    tens : tf.Tensor
        The 3D tensor of which to save slices.
    axis : int, optional
        The axis of the 3D tensor to slice along when saving slices.
        Default is `-1`.
    """

    if ext is None:
        ext = ''
    elif isinstance(ext, str) and ext[0] != '.':
        ext = '.' + ext
        ext = ext.lower()

    # Build indexer
    def get_indexer(shape, ax, slice_expr):
        if ax < 0:
            ax += len(shape)
        return np.index_exp[:] * ax + slice_expr + (np.index_exp[:] * (len(shape) - ax - 1))

    if ext == '.png' or ext == '.npy':
        if ext == '.png':
            arr = (tens*255).numpy().astype(np.uint8)
        else:
            arr = tens.numpy()
        for slice_no in range(arr.shape[axis]):
            fn = base_fn + f's{slice_no}' + ext
            if ext == '.png':
                imsave(fn, arr[get_indexer(arr.shape, axis, (slice_no,))])
            elif ext == '.npy':
                np.save(fn, arr[get_indexer(arr.shape, axis, (slice_no,))])
    else:
        raise ValueError(f'Invalid extension \'{ext}\'.')

def slice_and_save_npz(base_fn, axis=-1, ext='.npz', **tensors):
    """
    Slice and save a set of three-dimensional tensors and store them in .npz
    files.

    Parameters
    ----------
    base_fn : str
        The string to append to all files saved using this method.
    axis : int, optional
        The axis along which to slice. Default is -1, or the last axis.
    ext : str, optional
        The extension to append onto each file. Default is '.npz', but users
        may append additional extensions to the front of this (or change the
        extension entirely) if they wish.
    **tensors
        Series of keyword arguments of the form name=tensor, where tensor is
        an object of type tf.Tensor.
    """
    # Save tensors as npz files.
    # Need to assert tensors have the same shapes later.

    if not isinstance(base_fn, str):
        base_fn = str(base_fn)

    # Build indexer
    def get_indexer(shape, ax, slice_expr):
        if ax < 0:
            ax += len(shape)
        return np.index_exp[:] * ax + slice_expr + (np.index_exp[:] * (len(shape) - ax - 1))

    # Assuming dictionary key/value order remains consistent internally.
    arr_names = list(tensors.keys())
    arr_values = [tensor.numpy() for tensor in tensors.values()]
    for slice_no in range(arr_values[0].shape[axis]):
        fn = '_'.join([base_fn, f's{slice_no}{ext}'])
        slice_collection = {name: arr[get_indexer(arr.shape, axis, (slice_no,))]
            for name, arr in zip(arr_names, arr_values)}
        np.savez(fn, **slice_collection)

def save_3d_npz(vol_seg_pairs, out_path, class_names, vol_name, seg_name,
    no_slice=False):
    """
    Save pairs of image volumes and segmentation maps in a format useable
    by the SynthMorph training scripts.

    Parameters
    ----------
    vol_seg_pairs : list of tuple
        A collection of tuples, each with two tensors: (vol_tensor, lbl_tensor).
    out_path : str
        The base directory where files will be saved.
    class_names : list of str
        The names given to the subdirectories in which each class will be
        stored.
    vol_name : str
        Base name given to .npz files holding volume information.
    seg_name : str
        Base name given to .npz files holding segmentation map information.
    no_slice : bool, optional
        Whether or not to store the volume/segmentation maps as volumes or
        slices. Default `False`.

    """
        
    # Light checking
    if not isinstance(out_path, Path):
        out_path = Path(out_path)

    # Make the output directories.
    for name in class_names:
        makedirs(out_path / name, exist_ok=True)

    # Save the slices (or whole 3D images) to disk.
    for (img, lbl_map), name in zip(
            vol_seg_pairs,
            class_names):

        base_path = out_path / name

        # If no slicing, save the 3D structures; otherwise, save the 2D slices
        # along the last axis.
        if no_slice:
            img_path = base_path / vol_name
            seg_path = base_path / seg_name
            np.savez(img_path, vol=img, seg=lbl_map)
            np.savez(seg_path, vol=img, seg=lbl_map)
            
        else:
            vol_base, vol_ext = vol_name.split('.', 1)
            seg_base, seg_ext = seg_name.split('.', 1)
            slice_and_save_npz(base_path / vol_base, ext=vol_ext, axis=-1, vol=img, seg=lbl_map)
            slice_and_save_npz(base_path / seg_base, ext=seg_ext, axis=-1, vol=img, seg=lbl_map)

# load average deformation field in current folder
def load_avg_dense(folder):
    dense_files = [t.path for t in scandir(folder) if not t.is_dir() and t.name.startswith('dense')]
    return np.mean(np.stack([np.load(fp) for fp in dense_files], axis=0), axis=0)
  
def download_file(url, out_fp=None, return_response=False):
    """
    Download a file at a given url.

    Parameters
    ----------
    url : str
        The URL of the file.
    out_fp : str or None
        The path to which the file should be saved. If None, do not save. Default
        is None.
    return_response : bool
        Whether to return the content of the response as an object. If False,
        nothing is returned. Default is False.
    """
    with requests.get(url, stream=True) as response:
        if out_fp is not None:
            with open(out_fp, 'wb') as f:
                f.write(response.content)
        if return_response: return response.content

## INTRODUCED IN V4

# hypermorph network and model i/o
# provide specific name overrides so this can be used by other functions
def generate_tensorspec(tensor_collection: Collection[tf.Tensor],
                        name: str = 'collection') -> Collection[tf.TensorSpec]:
    """
    Converts a list of tf.Tensors to TensorSpecs.

    Parameters
    ----------
    tensor_collection : collection of tf.Tensors
        Collection of tensors to be converted into TensorSpecs.
    name : str
        Base name for TensorSpec objects in the collection.
    
    Returns
    -------
    Collection of TensorSpec objects
        TensorSpec collection that mirrors the types in tensor_collection.
    """

    if '__len__' not in dir(tensor_collection):
        raise ValueError('tensor_collection and its elements must be tf.Tensor objects '
                         'or enumerable collections.')
    return [tf.TensorSpec.from_tensor(t, name=name) # should automatically extract tensor names unless kwarg set
            if isinstance(t, tf.Tensor) else generate_tensorspec(t, name='%s_%d' % (name, i))
            for i, t in enumerate(tensor_collection)]

def np_save_fp(file: Union[str, int], *args, **kwargs):
    """
    np.save with a returned filepath. Mimics the np.save function and returns
    the filepath. See np.save for more information.
    """

    np.save(file, *args, **kwargs)
    return file

def pil_save_fp(img: np.ndarray, fp: str, *args, **kwargs):
    """
    np.save with a returned filepath. Mimics the np.save function and returns
    the filepath. See np.save for more information.
    """

    img.save(fp, *args, **kwargs)
    return fp

def preproc_fn(input, depth=None):
    """
    A preprocessing function that takes an input dictionary and normalizes/
    preprocesses all the values to align with the specifications called for
    by models.

    Images are normalized to [0, 1] and labels are one-hot encoded using either
    an inferred or specified depth.

    Parameters
    ----------
    input : dict
        Dictionary containing tensors to preprocess.
    depth : int or None
        The depth of the one-hot-encoded labelmaps.

    Returns
    -------
    output : dict
        A dictionary containing the preprocessed versions of the tensors in
        the input dictionary.
    """

    output = {}
    for nm, val in input.items():
        if nm.endswith('_img'): output[nm] = normalize(val,
                                                       min_value=0,
                                                       max_value=1,
                                                       lib=tf.experimental.numpy)
        elif nm.endswith('_lbl'): output[nm] = tf.one_hot(indices=tf.cast(val, tf.int32),
                                                          depth=depth if depth is not None
                                                                else tf.math.reduce_max(val) + 1,
                                                          dtype=tf.float32)
        else: output[nm] = val
    return output

def map2vxm(input, hparam_default=0.0):
    """
    Map inputs from a dictionary to a format compatible with the input of a model:

    ```
    [input, expected_output, sample_weights] where

    input = [moving_img, fixed_img, hparams, moving_lbl]
    expected_output = [fixed_img, 0]
    sample_weights = [1, hparams, 1]
    ```

    If 'hparams' is not included in the dataset, the associated parameter
    will be set to 'hparam_default'. This means the model will behave as
    if any corresponding loss/other coefficients are OMITTED (since hparam_default
    is 0) unless hparam_default is changed. This is for backwards-compatibility
    purposes. Use augment_dataset to set custom hparam values.

    Parameters
    ----------
    input : dict
        The dictionary containing relevant tensor values.
    include_lbl : bool
        Whether to include labels in the returned structure. If not, return 
        a zero tensor for the moving label. Default is True.
    
    Returns
    -------
    tuple
        Tensors arranged in the appropriate format.
    """

    return ((input['moving_img'],
             input['fixed_img'],
             input.get('hparams', tf.constant((hparam_default,))),
             input['moving_lbl']),
            (input['fixed_img'],
             tf.zeros((1,)),
             input['fixed_lbl']),
            (tf.ones((1,)),
             input.get('hparams', tf.constant((hparam_default,))),
             tf.ones((1,)),))

def load_dict_dataset(filepath,
                      example_description=EXAMPLE_DESCRIPTION,
                      out_types=EXAMPLE_OUT_TYPES,
                      out_shapes=EXAMPLE_OUT_SHAPES):
    """
    Load a TFRecord dataset in a format that is compatible with training a model.

    Parameters
    ----------
    filepath : str
        Filepath where the TFRecord is stored.
    depth : int or None
        Depth to use when creating one-hot-encoded labelmaps. Default is None,
        which infers the size from the file (might error for labelmaps of varying
        depth).
    example_description : dict or None
        A dictionary containing key-value pairs of the form `name: tf.io.Feature (FixedLenFeature, etc.)`.
        This will be useful for parsing the serialized Example information. Default value is supplied
        by utils.constants.
    out_types : dict or None
        A dictionary containing key-value pairs of the form `name: tf.dtype`. These will be used to convert
        arguments into their corresponding Tensorflow types. Default value is supplied
        by utils.constants.
    out_shapes : dict or None
        A dictionary containing key-value pairs of the form `name: tuple`. These will be used to make
        shape information available if tensors are used in Graph mode. Default value is supplied
        by utils.constants.
    
    Returns
    -------
    tf.data.Dataset
        A dataset consisting of returned dictionary elements. No preprocessing is applied.
    """

    # load serialized
    dsser = tf.data.TFRecordDataset(filepath) 

    # deserialize and make types uniform
    ds = dsser.map(lambda x: serialized_example_to_tensors(x,
                                                           example_description,
                                                           out_types,
                                                           out_shapes))
    return ds

def standardize_dtypes(tdict):
    """
    Standardize datatypes of dataset.
    """

    return {nm: tf.cast(val,
                        dtype=tf.int32 if nm.endswith('_lbl')
                                       else tf.float32)
            for nm, val in tdict.items()}

def load_dataset(filepath,
                 depth=None,
                 example_description=EXAMPLE_DESCRIPTION,
                 out_types=EXAMPLE_OUT_TYPES,
                 out_shapes=EXAMPLE_OUT_SHAPES):
    """
    Load a TFRecord dataset in a format that is compatible with training a model.

    Parameters
    ----------
    filepath : str
        Filepath where the TFRecord is stored.
    depth : int or None
        Depth to use when creating one-hot-encoded labelmaps. Default is None,
        which infers the size from the file (might error for labelmaps of varying
        depth).
    example_description : dict or None
        A dictionary containing key-value pairs of the form `name: tf.io.Feature (FixedLenFeature, etc.)`.
        This will be useful for parsing the serialized Example information. Default value is supplied
        by utils.constants.
    out_types : dict or None
        A dictionary containing key-value pairs of the form `name: tf.dtype`. These will be used to convert
        arguments into their corresponding Tensorflow types. Default value is supplied
        by utils.constants.
    out_shapes : dict or None
        A dictionary containing key-value pairs of the form `name: tuple`. These will be used to make
        shape information available if tensors are used in Graph mode. Default value is supplied
        by utils.constants.
    
    Returns
    -------
    tf.data.Dataset
        A dataset consisting of elements mapped to Hypermorph networks.
    """
    # Preprocess and return
    ds = load_dict_dataset(filepath=filepath,
                           example_description=example_description,
                           out_types=out_types,
                           out_shapes=out_shapes)
    dspre = ds.map(standardize_dtypes)
    dspre = dspre.map(lambda x: preproc_fn(x, depth=depth))
    dspre = dspre.map(map2vxm)
    return dspre

def augment_hparam_dataset(dataset, hp):
    """
    Augment a Hypermorph-capable dataset by changing the hyperparameter
    values associated with it.

    Parameters
    ----------
    dataset : tf.data.Dataset
        A dataset containing examples in the format output by `map2vxm`.
    hp : numeric or scipy.stats distribution
        A number to use for all hyperparameter values or a random distribution
        that will be sampled to generate new hyperparameter values.
    """
    def gen_fn():
        for datum in dataset:
            input, output, weight = [list(elem) for elem in datum]
            hp_shape = input[2].shape
            curr_vals = hp.rvs(size=hp_shape) if 'rvs' in dir(hp) else hp
            input[2] = tf.ones(input[2].shape, dtype=input[2].dtype) * tf.convert_to_tensor(curr_vals, dtype=input[2].dtype)
            weight[1] = tf.ones(weight[1].shape, dtype=weight[1].dtype) * tf.convert_to_tensor(curr_vals, dtype=weight[1].dtype)
            yield tuple([tuple(elem) for elem in [input, output, weight]])
    return tf.data.Dataset.from_generator(gen_fn, output_signature=dataset.element_spec)
