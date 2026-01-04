"""
MODELS

Contains all models and associated model helper methods.

Helper functions will be moved to appropriate locations with the release of v4.

Author: Peter Lais
Last updated: 10/15/2022
"""

from typing import Callable, Collection, List, Optional, Union
import warnings
import tensorflow as tf
import neurite as ne
import numpy as np
import voxelmorph as vxm
from tensorflow.keras import layers as KL, \
                             initializers as KI, \
                             Model

from .utils import (tensors_to_tensorspec,
                    sig2tensors,
                    tensorspec_to_dict,
                    tensors_to_dict,
                    maybe_rename_tensors,
                    maybe_remap_names_tensorspecs,
                    map_dict_keys)

#################
# MODEL CLASSES #
#################

class EnrichedFunctionalModel(Model):
    """
    An extension of the functional keras Model API to allow for the requesting of
    input and output signatures that can be overridden by subclasses.
    
    You should be able to use this method exactly as you would use keras.Model.
    The only difference is that you will now be able to call some additional
    methods from any model constructed using this class.
    """
    
    # get the signature of the method and transform the model's inputs
    def __init__(self,
                 inputs: Collection[tf.Tensor],
                 outputs: Collection[tf.Tensor],
                 dict_inputs: bool = False,
                 dict_outputs: bool = False,
                 fill_missing_dict_inputs: bool = False,
                 outputs_base_name: Optional[str] = None,
                 outputs_name_map: Optional[dict] = None,
                 **kwargs):
        """
        Parameters
        ----------
        inputs : keras.Input or list of keras.Input
            A list of Keras tensors that define the input of the functional model.
        outputs : keras.Input or list of keras.Input
            A list of Keras tensors that are connected to the input tensors by a TensorFlow
            graph.
        dict_inputs : bool, default False
            Whether to accept model inputs in a dictionary form. Good for ease of use.
        dict_outputs : bool, default False
            Whether to output model outputs in a dictionary form. Good for ease of use.
        fill_missing_dict_inputs : bool, default False
            If dict_inputs is True, then this argument determines whether any arguments
            missing in the supplied input dictionary should be auto-filled. This is good
            if you have a model with a large number of inputs that can be omitted
            (filled with zeros), but otherwise it is best practice to leave this False.
        outputs_base_name : str or None
            By default, the output name of tensors (or keys if the output is a dictionary)
            are the name of the 
        """
        
        # Initialize above so properties are accessible
        super().__init__(inputs=inputs, outputs=outputs, **kwargs)
        
        # Process dict inputs/outputs
        self.dict_inputs = dict_inputs
        self.dict_outputs = dict_outputs
        self.fill_missing = fill_missing_dict_inputs
        self.dict_output_mapping = outputs_name_map
        
        # Process dict outputs
        if outputs_base_name is not None:
            self._output_named_tensor_list = None
            self._output_default_name = outputs_base_name
        else:
            self._output_named_tensor_list = self.outputs
            self._output_default_name = 'tensor'
        
    # Return the input signature with more arguments
    # 'private' helper method to prevent abuse
    def _get_input_signature(self, dict_mode: Optional[bool] = None,
                             custom_inputs: Collection[tf.Tensor] = None):
        inputs = self.inputs if custom_inputs is None else custom_inputs
        dict_mode = dict_mode if dict_mode is not None else self.dict_inputs
        return tensorspec_to_dict(tensorspec_list=tensors_to_tensorspec(tensor_list=inputs)) \
               if dict_mode else tensors_to_tensorspec(tensor_list=inputs)

    def get_input_signature(self):
        """
        Return the input signature for this model. Automatically converts to
        dictionary form if the model takes dictionary inputs.

        Returns
        -------
        list or dict
            The model input signature.
        """
        return self._get_input_signature(dict_mode=None,
                                         custom_inputs=None)
    
    # Set a default name if named tensor list (since base name was specified)
    # Return the output signature with more arguments
    # 'private' helper method to prevent abuse
    def _get_output_signature(self, dict_mode: Optional[bool] = None,
                              custom_outputs: Collection[tf.Tensor] = None):
        outputs = self.outputs if custom_outputs is None else custom_outputs
        dict_mode = dict_mode if dict_mode is not None else self.dict_outputs
        out_tensorspec = tensors_to_tensorspec(
            tensor_list=outputs,
            default_name=(self._output_default_name if self._output_named_tensor_list is None
                          else None))
        if dict_mode:
            out_tensorspec_dict = tensorspec_to_dict(tensorspec_list=out_tensorspec)
            if self.dict_output_mapping is not None:
                out_tensorspec_dict = map_dict_keys(input_dict=out_tensorspec_dict,
                                                    conversion_dict=self.dict_output_mapping)
            return out_tensorspec_dict
        elif self.dict_output_mapping is not None:
            return maybe_remap_names_tensorspecs(tensorspecs=out_tensorspec,
                                                 remap_dict=self.dict_output_mapping)
        else: return out_tensorspec
    
    def get_output_signature(self):
        """
        Return the output signature for this model. Automatically converts to
        dictionary form if the model takes dictionary outputs.

        Returns
        -------
        list or dict
            The model output signature.
        """
        return self._get_output_signature(dict_mode=None,
                                          custom_outputs=None)
        
    def call(self, inputs: Collection[tf.Tensor], *, training: bool, **kwargs) -> Union[List[tf.Tensor], dict]:
        """
        Call the model. See keras.Model.call for details.
        """
        assert not (isinstance(inputs, dict) ^ self.dict_inputs), \
            'Expected a %s input, got %s.' % ('dict' if self.dict_inputs else 'non-dict',
                                              type(inputs).__name__)
        if self.dict_inputs: inputs = sig2tensors(signature=tensors_to_tensorspec(self.inputs),
                                                   tensor_dict=inputs,
                                                   fill_missing=self.fill_missing)
        result = super().call(inputs, **kwargs)
        if self.dict_outputs: 
            result = tensors_to_dict(tensor_list=result,
                                     named_tensor_list=self._output_named_tensor_list,
                                     default_name=self._output_default_name)
            if self.dict_output_mapping is not None:
                result = map_dict_keys(input_dict=result,
                                       conversion_dict=self.dict_output_mapping)
        elif self.dict_output_mapping is not None:
            result = maybe_rename_tensors(tensors=result,
                                          names=self._get_output_signature(dict_mode=False))
        return result

    # def call_seeded(self, inputs, *, training, **kwargs):
    #     raise NotImplementedError('This model has no parameters to seed.')


class SynthMorph(EnrichedFunctionalModel):
  """
  Siamese for the first portion of it, followed by a traditional UNet.
  """

  class SiameseInput(Model):
    """
    A collection of KL that intakes a list of two images, concatenates them, and converts them into
    an output shape. The output has the same shape as each of the inputs.
    """

    def __init__(
      self,
      input_shape: Collection[int],
      activation: Union[str, Callable] = 'relu',
      name: str = 'siamese_in',
      kernel_initializer: Union[str, Callable] = 'glorot_uniform',
      bias_initializer: Union[str, Callable] = 'zeros'
    ):

      # Make the inputs and outputs.
      src_image = KL.Input(shape=input_shape, name='%s_src_image' % name)
      tgt_image = KL.Input(shape=input_shape, name='%s_tgt_image' % name)
      inputs = [src_image, tgt_image]
      num_filters = input_shape[-1]

      # Convolve each layer separately.
      inputs_conv = []
      for input_ in inputs:
        inputs_conv.append(
          KL.Conv2D(
            filters=num_filters,
            kernel_size=3,
            padding='same',
            name='%s_indep_conv' % input_.name,
            activation=activation,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer
          )(input_)
        )

      # Concatenate the inputs and downconvolve.
      input_stack = KL.Concatenate(
        axis=-1,
        name='%s_cat' % name
      )(inputs_conv)
      outputs = KL.Conv2D(
        filters=num_filters,
        kernel_size=3,
        padding='same',
        name='%s_joint_conv' % name,
        activation=activation,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer
      )(input_stack)
    
      # Make the model.
      super().__init__(name=name, inputs=inputs, outputs=outputs)

  class BaseUNet(Model):
    """
    A simple UNet without the final 1x1 conv layer at the end.
    """

    def __init__(
      self,
      input_shape: Collection[int],
      activation: Union[str, Callable] = 'relu',
      name: str = 'unet',
      kernel_initializer: Union[str, Callable] = 'glorot_uniform',
      bias_initializer: Union[str, Callable] = 'zeros'
    ):

      input_ = KL.Input(
        shape=input_shape,
        name='%s_input' % name
      )

      # Make generators.
      def create_conv(filter_ct, name):
        return KL.Conv2D(
          filters=filter_ct,
          kernel_size=3,
          padding='same',
          activation=activation,
          name=name,
          kernel_initializer=kernel_initializer,
          bias_initializer=bias_initializer
        )
      def create_maxpool(name):
        return KL.MaxPool2D(
          pool_size=2, strides=2,
          name=name
        )
      def create_convtranspose(filter_ct, name):
        return KL.Conv2DTranspose(
          filters=filter_ct,
          kernel_size=2,
          strides=2,
          padding='same',
          name=name,
          kernel_initializer=kernel_initializer,
          bias_initializer=bias_initializer
        )
      
      # Make the down stack.
      down_filter_cts = [64, 128, 256, 512, 1024]
      skips = []
      for ct in down_filter_cts:
        x = create_maxpool(
          name='%s_maxpool_down_%d' % (name, ct)
        )(skips[-1]) if skips else input_
        for i in range(2):
          x = create_conv(
            filter_ct=ct,
            name='%s_conv_down_%d_%d' % (name, ct, i)
          )(x)
        skips.append(x)
      
      # Make the up stack.
      x = skips.pop()
      # lo_dim_x = x # cache?
      for ct, skip in reversed(list(zip(down_filter_cts[:-1], skips))):
        x = create_convtranspose(
          filter_ct=ct,
          name='%s_upconv_%d' % (name, ct)
        )(x)
        x = KL.Concatenate(
          axis=-1,
          name='%s_cat_%d' % (name, ct)
        )([x, skip])
        for i in range(2):
          x = create_conv(
            filter_ct=ct,
            name='%s_conv_up_%d_%d' % (name, ct, i)
          )(x)
      
      outputs = x
      super().__init__(inputs=input_, outputs=outputs, name=name)

  class SVFFormer(Model):
    """
    Takes the output of a UNet and turns it into a 2D SVF.
    """

    def __init__(
      self,
      input_shape: Collection[int],
      activation: Union[str, Callable] = 'relu',
      name: str = 'def_field_former',
      kernel_initializer: Union[str, Callable] = 'glorot_uniform',
      bias_initializer: Union[str, Callable] = 'zeros'
    ):

      # Make the input and the output shape. Also extract relevant information.
      input_ = KL.Input(
        shape=input_shape,
        name='%s_input' % name
      )

      # Make the 2D deformation map.
      output = KL.Conv2D(
        filters=2,
        kernel_size=2,
        padding='same',
        activation=activation,
        name='%s_conv' % name,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer
      )(input_)

      super().__init__(inputs=input_, outputs=output, name=name)

  # Finalize this later.
  def __init__(
    self,
    input_shape: Collection[int],
    name: str = 'sm',
    siamese_kernel_initializer: Union[str, Callable] ='glorot_uniform',
    siamese_bias_initializer: Union[str, Callable] ='zeros',
    siamese_activation: Union[str, Callable] ='relu',
    baseunet_kernel_initializer: Union[str, Callable] ='glorot_uniform',
    baseunet_bias_initializer: Union[str, Callable] ='zeros',
    baseunet_activation: Union[str, Callable] ='relu',
    svfformer_kernel_initializer: Union[str, Callable] ='glorot_uniform',
    svfformer_bias_initializer: Union[str, Callable] ='zeros',
    svfformer_activation: Union[str, Callable, None] =None,
    auxiliary_outputs: Collection[str] = [],
    **kwargs
  ):

    # Form the inputs.
    input_image_src = KL.Input(
      shape=input_shape,
      name='%s_moving_img' % name
    )
    input_image_tgt = KL.Input(
      shape=input_shape,
      name='%s_fixed_img' % name
    )
    inputs = [input_image_src, input_image_tgt]

    # Make the Siamese input layer.
    siamese_output = self.SiameseInput(
      input_shape=input_shape,
      activation=siamese_activation,
      name='%s_siamese_in' % name,
      kernel_initializer=siamese_kernel_initializer,
      bias_initializer=siamese_bias_initializer
    )([input_image_src, input_image_tgt])

    # Run the image through the main model.
    unet_output = self.BaseUNet(
      input_shape=siamese_output.shape[1:],
      activation=baseunet_activation,
      name='%s_unet' % name,
      kernel_initializer=baseunet_kernel_initializer,
      bias_initializer=baseunet_bias_initializer
    )(siamese_output)

    # Turn the output into an SVF.
    svf_output = self.SVFFormer(
      input_shape=unet_output.shape[1:],
      activation=svfformer_activation,
      name='%s_def_field_former' % name,
      kernel_initializer=svfformer_kernel_initializer,
      bias_initializer=svfformer_bias_initializer
    )(unet_output)

    # Integrate via a vxm.layers.VecInt layer.
    def_output = vxm.layers.VecInt(
      indexing='ij',
      method='ss',
      int_steps=7,
      out_time_pt=1,
      name='%s_vec_int' % name
    )(svf_output)
    
    # Determine the inverse via a vxm.layers.VecInt layer.
    inv_def_output = vxm.layers.VecInt(
      indexing='ij',
      method='ss',
      int_steps=7,
      out_time_pt=1,
      name='%s_inv_vec_int' % name
    )(-svf_output)
    
    # Finally, transform the image with a vxm.layers.SpatialTransformer.
    output = vxm.layers.SpatialTransformer(
      interp_method='linear',
      indexing='ij',
      single_transform=False,
      fill_value=None,
      shift_center=True,
      name='%s_spatial_transform' % name
    )([input_image_src, def_output])
    outputs = [output]

    # Make auxiliary outputs available to wrapped classes and such.
    self.references = ne.modelio.LoadableModel.ReferenceContainer()
    self.references.siamese_output = siamese_output
    self.references.unet_output = unet_output
    self.references.svf_output = svf_output
    self.references.def_output = def_output
    self.references.inv_def_output = inv_def_output

    # Include some auxiliary outputs as actual outputs
    for aux_out in auxiliary_outputs:
      outputs.append(getattr(self.references, aux_out))

    super().__init__(inputs=inputs, outputs=outputs, name=name, **kwargs)


class STaCkerSemiSupervised(EnrichedFunctionalModel):
  """
  STACKER model
  """

  def __init__(
    self,
    img_shape: Collection[int],
    lbl_shape: Collection[int],
    name: str = 'sm_semi',
    input_names: Optional[Collection[str]] = None,
    output_names: Optional[Collection[str]] = None,
    efm_kwargs: dict = {},
    **kwargs
  ):

    # Base model that will give us a segmentation map.
    base_model = SynthMorph(
      input_shape=img_shape,
      name='%s_base' % name,
      **kwargs,
      **efm_kwargs
    )

    # Form the inputs. Draw from the inputs of UNet2D_V4 since they're
    # already taken from there.
    input_label_src = KL.Input(
      shape=lbl_shape,
      name='moving_lbl'
    )
    base_inputs = base_model.inputs
    inputs = base_inputs + [input_label_src]

    # Form the outputs, using the def_output to additionally deform
    # the passed in label image.
    base_outputs = base_model.outputs
    def_output = base_model.references.def_output
    lbl_image_transformed = vxm.layers.SpatialTransformer(
      interp_method='linear',
      indexing='ij',
      single_transform=False,
      fill_value=None,
      shift_center=True,
      name='%s_spatial_transform' % name
    )([input_label_src, def_output])
    outputs = base_outputs + [lbl_image_transformed]

    # Auxiliary outputs. base_model will contain most references already in
    # its own references property (alias provided for convenience)
    self.references = ne.modelio.LoadableModel.ReferenceContainer()
    self.references.base_model = base_model
    self.references.base_model_refs = base_model.references
    
    # Name the inputs and outputs if desired. Makes it easier to map
    # losses and metrics using dicts if you know the names.
    # Otherwise output names are output_1, output_2, ... by default,
    # not sure if similar for inputs.
    # https://datascience.stackexchange.com/questions/74406/custom-output-names-for-keras-model
    # https://stackoverflow.com/questions/54750890/multiple-metrics-to-specific-inputs
    if input_names is not None:
      inputs = {nname: input for nname, input in zip(input_names, inputs)}
    if output_names is not None:
      outputs = {nname: output for nname, output in zip(output_names, outputs)}
   
    # Make a Model that adheres to these specifications.
    super().__init__(inputs=inputs, outputs=outputs, name=name, **efm_kwargs)


