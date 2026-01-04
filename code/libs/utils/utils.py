"""
UTILS

Functions for general library functionality.

Author: Peter Lais
Last updated: 09/16/2023
"""

# Label generation functions
from typing import Collection, List, Literal, Optional, Tuple
import numpy as np
from scipy.stats import loguniform
import tensorflow as tf
import numpy as np
from importlib import import_module # needed for V1 functions

## INTRODUCED IN V1
# Legacy inclusions
# These may seem unneccesary but are needed if users want to import them.
# Should we remove these? Check your code to see if these are used.

def remove_singletons(*args, out_lens=2):
    """
    A utility method meant to replace singleton values with arrays of size `out_lens`.

    Parameters
    ----------
    args : array of arguments
        Array of arguments that may contain singleton values. All arguments are returned
        as-is unless any singleton values are present, in which the singleton is replaced
        by a list of length `out_lens` filled by that singleton value.
    out_lens : int
        See above; the size to use when replacing singletons with lists.
    
    Returns
    -------
    list
        All arguments with singletons replaced with lists.
    """

    out = []
    iterable = zip(args, [out_lens] * len(args)) if '__len__' not in dir(out_lens) \
                                                else zip(args, out_lens)
    for arg, out_len in iterable:
        if '__len__' in dir(arg):
            if isinstance(arg, np.ndarray) and arg.size == 1: arg = arg.item()
            elif len(arg) == 1: arg = arg[0]
        if '__len__' not in dir(arg) and out_lens != 1: out.append([arg] * out_len)
        else: out.append(arg)
    return out[0] if len(out) == 1 else out

def random_hparam(dist, *args, **kwargs):
    """
    Given a distribution dist, sample it.

    Parameters
    ----------
    dist : scipy.stats distribution.
    args : positional arguments to dist.
    kwargs : keyword arguments to dist.

    Returns
    -------
    Output of sampling dist.
    """
    assert 'rvs' in dir(dist), 'dist must conform to scipy-like standards (rvs function).'
    return dist.rvs(*args, **kwargs)
    
def normalize(*args, min_value=0, max_value=1, lib=np): # arguments are np
    """
    Function to use in normalizing values present in args.

    Parameters
    ----------
    args : list of np.ndarray, tf.Tensor, or similar
        List of arguments whose values must be normalized.
    min_value : float
        Minimum value of normalized arrays. Default is 0.
    max_value : float
        Maximum value of normalized arrays. Default is 1.
    lib : module or str
        Module to use when normalizing. Useful when you're attempting to normalize
        tensors since tf.Tensors don't always play nice when numpy functions are
        called on them.
    """

    # get the numpy-like lib and do the op
    if isinstance(lib, str): lib = import_module(lib)
    out = [(arg - lib.amin(arg)) / (lib.amax(arg) - lib.amin(arg))
                * (max_value - min_value) + min_value
            if lib.amin(arg) != lib.amax(arg) else lib.ones_like(arg) * min_value for arg in args]
    return out[0] if len(out) == 1 else out

## INTRODUCED IN V4

# Convert the deformation maps into magnitudes and angles.
# Alias for normalize but a tensorflow function. Needs the decoration.
@tf.function
def minmax(tens: tf.Tensor) -> tf.Tensor:
    """
    Minmax-scale tens to range from [0,1].

    Parameters
    ----------
    tens : tf.Tensor-like
        An object that behaves as a Tensor.
    
    Returns
    -------
    tf.Tensor-like
        Output minmax-scaled tensor.
    """
    tens = tf.cast(tens, tf.float32)
    maxx = tf.math.reduce_max(tens)
    minn = tf.math.reduce_min(tens)
    return (tens - minn) / (maxx - minn)

def tensor_to_tensorspec(tensor: tf.Tensor, default_name: str = None) -> tf.TensorSpec:
    """
    Try to convert a single tensor-like element to a TensorSpec.
    
    Parameters
    ----------
    tensor : tf.Tensor-like
        An object that behaves as a Tensor.
    default_name : str or None
        A name that is applied to the generated TensorSpec in lieu of the tensor-like object's
        name.
        
    Returns
    -------
    tf.TensorSpec
        Output TensorSpec.
    """
    
    if isinstance(tensor, tf.Tensor):
        return tf.TensorSpec.from_tensor(tensor)
    else:
        try: return tf.TensorSpec(shape=tensor.shape, dtype=tensor.dtype, name=(tensor.name if default_name is None
                                                                                            else default_name))
        except Exception as e: raise Exception(
            'The passed-in element does not behave as a Tensor.') from e

def is_list_or_tuple(*structs):
    """
    Determine if an argument (or series thereof) is a list or tuple.
    
    Parameters
    ----------
    structs : list of arguments
        Arguments to check if list or tuple.
    
    Returns
    -------
    bool or list of bool
        If a single struct is supplied, then this will be a single boolean. If it is a list,
        then a list of booleans will be returned.
    """
    out = [isinstance(struct, (list, tuple)) for struct in structs]
    return out[0] if len(out) == 1 else out

def tensors_to_tensorspec(tensor_list: Collection[tf.Tensor],
                          default_name: Optional[str] = None) -> List[tf.TensorSpec]:
    """
    Try to convert a list of tensor-like elements to TensorSpec objects.
    
    Parameters
    ----------
    tensor_list : tf.Tensor-like
        A list of objects that behave as Tensor objects.
    default_name : str or None
        A name that is applied to the generated TensorSpec objects in lieu of using
        the name of each tensor-like object.
        
    Returns
    -------
    list of tf.TensorSpec
        Output TensorSpec list.
    """
    if not is_list_or_tuple(tensor_list): return tensor_to_tensorspec(tensor=tensor_list, default_name=default_name)
    return tuple([tensor_to_tensorspec(tensor=t, default_name=(None if default_name is None
                                                               else '%s_%d' % (default_name, i)))
                  if not is_list_or_tuple(t) else tensors_to_tensorspec(tensor_list=t,
                                                                        default_name=(None if default_name is None
                                                                                      else '%s_%d' % (default_name, i)))
                  for i, t in enumerate(tensor_list)])

def sig2tensors(signature: Collection[tf.TensorSpec], tensor_dict: dict,
                fill_missing: bool = False) -> List[tf.Tensor]:
    """
    Convert a signature (a collection of TensorSpec objects) to a collection of
    Tensors as specified by a dictionary tensor_dict.

    If we have a TensorSpec element, return its corresponding tensor in the dictionary.
    If the element is not a TensorSpec, we assume it is a list and iterate through that
    recursively.

    Parameters
    ----------
    signature : list of TensorSpec
        A list of TensorSpec objects that will form the output colleciton of tensors.
    tensor_dict : dict
        A dictionary of string-tensor pairs that will be used to draw from signature.

    Returns
    -------
    list of tf.Tensor
        Output collection of tensors that subscribes to the format specified in signature.
    """
    if not fill_missing:
        return tuple([tensor_dict[ts.name] if isinstance(ts, tf.TensorSpec)
            else sig2tensors(ts, tensor_dict, fill_missing) for ts in signature])
    else:
        # Return filled-in values with zeros in case they're needed.
        return tuple([tensor_dict.get(ts.name, tf.zeros((next(iter(tensor_dict.values())).shape[0],) + ts.shape))
            if isinstance(ts, tf.TensorSpec)
            else sig2tensors(ts, tensor_dict, fill_missing) for ts in signature])

def tensorspec_to_dict(tensorspec_list: Collection[tf.TensorSpec],
                       default_name: str = 'tensor') -> dict:
    """
    Convert a collection of Tensorspec objects to a dictionary mapping strings
    to Tensorspec objects.

    Parameters
    ----------
    tensorspec_list : list of tf.TensorSpec
        List of TensorSpecs to convert.
    default_name : str or None
        A name that is applied to the generated TensorSpec objects in lieu of using
        the name of each tensorspec-like object.
    
    Returns
    -------
    dict
        A dictionary of string-TensorSpec pairs that maps the names of
        the input TensorSpec objects to themselves.
    """
    if not is_list_or_tuple(tensorspec_list): tensor_list = [tensor_list]        
    output_dict = {}
    for i, tensorspec in enumerate(tensorspec_list):
        if is_list_or_tuple(tensorspec):
            name = '%s_%d' % (default_name, i)
            to_add = (tensorspec if not is_list_or_tuple(tensorspec)
                             else tensorspec_to_dict(tensorspec_list=tensorspec,
                                                     default_name='%s_%d' % (default_name, i)))
            output_dict[name] = to_add
        else:
            try: output_dict[tensorspec.name] = tensorspec
            except AttributeError:
                output_dict['%s_%d' % (default_name, i)] = tensorspec
    return output_dict
    
def tensors_to_dict(tensor_list: Collection[tf.Tensor], named_tensor_list: Optional[Collection[str]] = None,
                    default_name: str = 'tensor') -> dict:
    """
    Convert a collection of Tensor objects to a dictionary mapping strings
    to Tensor objects.

    Parameters
    ----------
    tensor_list : list of tf.Tensor
        List of Tensors to convert.
    named_tensor_list : list of tf.Tensor
        Names to give the tensors in the output dictionary.
    default_name : str or None
        A name that is applied to the generated Tensor objects in lieu of using
        the name of each tensor-like object.
    
    Returns
    -------
    dict
        A dictionary of string-Tensor pairs that maps the names of
        the input Tensor objects to themselves.
    """
    if not is_list_or_tuple(tensor_list): tensor_list = [tensor_list]
    if named_tensor_list is not None and not is_list_or_tuple(named_tensor_list):
        named_tensor_list = [named_tensor_list]
    assert named_tensor_list is None or len(tensor_list) == len(named_tensor_list), \
        'Named tensor list must be the same length as tensor list. ' \
        '(Former: %d, latter: %d)' % (len(named_tensor_list), len(tensor_list))
        
    output_dict = {}
    enumerable = tensor_list if named_tensor_list is None else zip(tensor_list, named_tensor_list)
    for i, tensor in enumerate(enumerable):
        if named_tensor_list is not None: tensor, named_tensor = tensor
        starting_name = '%s_%d' % (default_name, i) if default_name is not None else tensor.name
        if is_list_or_tuple(tensor):
            to_add = (tensor if not is_list_or_tuple(tensor)
                             else tensors_to_dict(tensor_list=tensor,
                                                  named_tensor_list=named_tensor,
                                                  default_name='%s_%d' % (default_name, i)))
            output_dict[starting_name] = to_add
        else:
            try: output_dict[starting_name if named_tensor_list is None
                             else named_tensor.name] = tensor
            except AttributeError:
                output_dict[starting_name] = tensor
    return output_dict
    
def maybe_rename_tensors(tensors: Collection[tf.Tensor],
                         names: Collection[str]) -> Tuple[tf.Tensor]:
    """
    Maybe rename a collection of tensors with the given names. If an object
    cannot be renamed, it is simply returned.

    Parameters
    ----------
    tensors : list of tf.Tensor or alike objects
        A list of tensors to rename.
    names : list of str
        A list of strings of the same length as tensors.

    Returns
    -------
    list of tf.Tensor
        A list of renamed tensors.
    """
    if not is_list_or_tuple(tensors): return maybe_rename_tensor(tensor=tensors, name=names)
    return tuple([maybe_rename_tensor(tensor=t, name=n) if not is_list_or_tuple(t) else 
                  maybe_rename_tensors(tensors=t, names=n) for t, n in zip(tensors, names)])

# rename tensorflow and possibly Keras tensors
# if not valid, just pass it back
def maybe_rename_tensor(tensor: tf.Tensor, name: str) -> tf.Tensor:
    """
    Maybe rename a tensor with the given names. If an object
    cannot be renamed, it is simply returned.

    Parameters
    ----------
    tensor : tf.Tensor or alike object
        The tensor to rename.
    name : str
        The intended name for the tensor.

    Returns
    -------
    tf.Tensor
        A renamed tensor.
    """
    if is_list_or_tuple(tensor) and len(tensor) == 1: tensor = tensor[0]
    if is_list_or_tuple(name) and len(name) == 1: name = name[0]
    assert not np.any(is_list_or_tuple(tensor, name)), 'Cannot pass in lists of tensors or names ' \
        'that have multiple items.'
    
    # make sure we have a string. Can pass tensorspecs
    if isinstance(name, tf.TensorSpec): name = name.name
    elif not isinstance(name, str): name = str(name)
        
    try: return tf.identity(tensor, name=name)
    except Exception: return tensor

def maybe_remap_names_tensorspecs(tensorspecs: Collection[tf.TensorSpec],
                                  remap_dict: dict) -> Tuple[tf.TensorSpec]:
    """
    Rename a collection of TensorSpec objects as specified by remap_dict.

    Parameters
    ----------
    tensorspecs : list of TensorSpec
        A list of TensorSpec objects that are to be renamed. If the name of a TensorSpec
        is not in remap_dict.keys(), it is not renamed.
    remap_dict : dict
        A dictionary of string-string pairs that will be used to rename TensorSpec objects.
        Each key in the dictionary is the source name that will be mapped to a target name
        as specified by its value: {source_name: tgt_name}

    Returns
    -------
    list of tf.TensorSpec
        Output collection of renamed TensorSpec objects.
    """
    if not is_list_or_tuple(tensorspecs): return maybe_rename_tensorspec(tensorspec=tensorspecs, name=remap_dict.get(tensorspecs.name, tensorspecs.name))
    return tuple([maybe_rename_tensorspec(tensorspec=t, name=remap_dict.get(t.name, t.name)) if not is_list_or_tuple(t) else 
                  maybe_remap_names_tensorspecs(tensorspecs=t, remap_dict=remap_dict) for t in tensorspecs])
    
def maybe_rename_tensorspecs(tensorspecs: Collection[tf.TensorSpec],
                             names: Collection[str]) -> Tuple[tf.TensorSpec]:
    """
    Maybe rename a collection of TensorSpec objects with the given names. If an object
    cannot be renamed, it is simply returned.

    Parameters
    ----------
    tensors : list of tf.TensorSpec or alike objects
        A list of TensorSpec or similar objects to rename.
    names : list of str
        A list of strings of the same length as tensorspecs.

    Returns
    -------
    list of tf.Tensor
        A list of renamed tensors.
    """
    if not is_list_or_tuple(tensorspecs): return maybe_rename_tensorspec(tensorspec=tensorspecs, name=names)
    return tuple([maybe_rename_tensorspec(tensorspec=t, name=n) if not is_list_or_tuple(t) else 
                  maybe_rename_tensorspecs(tensorspecs=t, names=n) for t, n in zip(tensorspecs, names)])
    
def maybe_rename_tensorspec(tensorspec: tf.TensorSpec, name: str) -> tf.TensorSpec:
    """
    Maybe rename a TensorSpec with the given name. If an object
    cannot be renamed, it is simply returned.

    Parameters
    ----------
    tensor : tf.TensorSpec or alike object
        The TensorSpec to rename.
    name : str
        The intended name for the TensorSpec.

    Returns
    -------
    tf.TensorSpec
        A renamed TensorSpec.
    """
    if is_list_or_tuple(tensorspec) and len(tensorspec) == 1: tensorspec = tensorspec[0]
    if is_list_or_tuple(name) and len(name) == 1: name = name[0]
    assert not np.any(is_list_or_tuple(tensorspec, name)), 'Cannot pass in lists of tensorspecs or names ' \
        'that have multiple items.'
    
    # make sure we have a string. Can pass tensorspecs
    if isinstance(name, tf.TensorSpec): name = name.name
    elif not isinstance(name, str): name = str(name)
        
    try: return tf.TensorSpec.from_spec(tensorspec, name=name)
    except Exception: return tensorspec
    
# soft mapping; anything not in conversion_dict is not mapped
def map_dict_keys(input_dict: dict, conversion_dict: dict) -> dict:
    """
    Rename a dictionary of tensors or TensorSpec objects using the source-target
    name pairs specified by conversion_dict.
    
    Parameters
    ----------
    input_dict : dict
        Input dictionary of string-tensor (or string-TensorSpec) objects to remap
        (meaning rename).
    conversion_dict : dict
        String-string dictionary consisting of corresponding source and target names.
        For example: {source_name: tgt_name}
        
    Returns
    -------
    dict
        A dictionary consisting of new string-tensor (or string-TensorSpec) pairs based
        on the name mapping specified by conversion_dict.
    """
    output_dict = {}
    for key, val in input_dict.items():
        name = conversion_dict.get(key, key)
        
        # bonus: try and rename tensorspecs or tensors if possible
        if isinstance(val, tf.TensorSpec): val = maybe_rename_tensorspec(tensorspec=val,
                                                                         name=name)
        else: val = maybe_rename_tensor(tensor=val, name=name)
            
        output_dict[name] = val
    return output_dict

# simplify the process of output remap generation
def simple_efm_kwargs_generator(dict_inputs: bool = True,
                                dict_outputs: bool = True,
                                outputs_base_name: str = 'tensor',
                                output_names_list: Optional[Collection[str]] = None) -> dict:
    """
    Simple EnrichedFunctionalModel keyword argument generator.
    
    This function is intended to provide typical keyword arguments fed into
    EnrichedFunctionalModels to control their output. Typical usage will
    appear as follows (assume `Model` is an `EnrichedFunctionalModel` derived
    class):

    ```
    model = Model(efm_kwargs=simple_efm_kwargs_generator(dict_inputs=True,
                                                         ...))
    ```

    Parameters
    ----------
    dict_inputs : bool
        Whether to indicate that the EnrichedFunctionalModel should have dictionary inputs.
    dict_outputs : bool
        Whether to indicate that the EnrichedFunctionalModel should have dictionary outputs.
    outputs_base_name : str
        The base name to assign to tensors being output from the model. If output_names_list
        is specified (not None), this will be auto-assigned to 'tensor' if not specified
        explicitly.
    output_names_list : list of str or None
        An optional list of names to give tensors output from the corresponding model.
    """
    if output_names_list is not None:
        # if output_names_list is specified, auto-assign a base name if not assigned.
        if outputs_base_name is not None: outputs_base_name = 'tensor'
        if not is_list_or_tuple(output_names_list): output_names_list = [output_names_list]
        output_remap_dict = {'%s_%d' % (outputs_base_name, i): tgt_nm
                                for i, tgt_nm in enumerate(output_names_list)}
    else: output_remap_dict = None
    return {'dict_inputs': dict_inputs,
            'dict_outputs': dict_outputs,
            'outputs_base_name': outputs_base_name,
            'outputs_name_map': output_remap_dict}

def assert_indexing(indexing: Literal['ij', 'xy']) -> str:
    """
    Asserts that the passed-in parameter is a valid indexing style.

    Parameters
    ----------
    indexing : 'ij' or 'xy'
        Indexing style. Row-major ('ij') or column-major ('xy').
    
    Returns
    -------
    indexing : 'ij' or 'xy'
        The passed-in indexing method.
    """
    indexing = indexing.lower()
    assert indexing in ('xy', 'ij'), \
        'Indexing method \'%s\' is not valid, must be (\'ij\' or \'xy\').' % indexing
    return indexing

class neg_loguniform():
    """
    A class that simply provides negative values corresponding to scipy's
    loguniform distribution. See `scipy.stats.loguniform` for details
    about initialization arguments.

    ONLY the `rvs` method should be used with this class. This is a shim.
    """
    def __init__(self, *args, **kwargs):
        self.loguniform = loguniform(*args, **kwargs)
    def rvs(self, *args, **kwargs):
        return -self.loguniform.rvs(*args, **kwargs)

class discretehp():
    """
    A class that may be sampled in a manner similar to scipy.stats distributions
    for random values in an array. This should be used with `synthimglbl` or
    `augment_hparam_dataset` to generate random parameters.
    """

    def __init__(self, values, seed=None):
        """
        Parameters
        ----------
        values : list-like
            A list of values from which to randomly sample.
        seed : int or None
            A random seed to use in seeding the random generator used internally
            by this class.
        """

        self._rng = np.random.default_rng(seed=seed)
        self._vals = values

    def rvs(self, size: int) -> np.ndarray:
        """
        Randomly sample values from the internal array of this class.

        Parameters
        ----------
        size : int or tuple
            A desired size/shape of output values used for this function.
        
        Returns
        -------
        Random variates in the shape of `size`.
        """

        # log-uniform distribution, uniformly choose among log-distributed samples
        return self._rng.choice(a=self._vals,
                                size=size)

