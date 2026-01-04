"""
CALLBACKS

Useful callback functions during model training.

Author: Peter Lais
Last updated: 10/15/2022
"""

from typing import Collection, Union
import tensorflow as tf
import tensorflow.keras.callbacks as KC
from .utils import normalize
from .utils.image.convert import def2img

class Writer(KC.Callback):
    """
    A class intended to write relevant information during model training to
    Tensorboard. ONLY COMPATIBLE WITH THE SYNTHMORPH MODELS FOR NOW.

    At the end of every epoch, this writes the following information to disk:

    * `moving_imgs`: A sample of unaligned moving image patches fed to the model.
    * `fixed_imgs`: A sample of unaligned fixed image patches fed to the model.
    * `moving_lbls`: A sample of unaligned moving label patches fed to the model.
    * `fixed_lbls`: A sample of unaligned fixed label patches fed to the model.
    * `moved_imgs`: A sample of aligned moved image patches output by the model.
    * `moved_lbls`: A sample of aligned moved label patches output by the model.
    * `mags`: An image showing the pixel-level magnitudes of the deformation field
              at any given region. Useful for analyzing if deformations are smooth.
    * `angles`: An image showing the pixel-level angles of the deformation field
                at any given region. Useful for analyzing if deformations are smooth,
                although regions may be sharp if the angle wraps around from high
                to low value [359 degrees to 0 degrees, for example].
    * `deffs`: A matplotlib quiver plot showcasing the deformation field. Color of
               the arrows correspond to magnitude, with higher-magnitude arrows
               getting 'yellower' colors.
    * `mags_max`: A scalar value showing the maximum deformation magnitude since
                  the above plots only show relative, and not absolute, values.
    * `mags_min`: A scalar value showing the minimum deformation magnitude since
                  the above plots only show relative, and not absolute, values.
    * `angles_max`: A scalar value showing the maximum deformation angle since
                    the above plots only show relative, and not absolute, values.
                    Given in radians.
    * `angles_min`: A scalar value showing the minimum deformation angle since
                    the above plots only show relative, and not absolute, values.
                    Given in radians.
    * `lr`: A scalar value showing the learning rate at the given epoch. Useful
            for observing if a learning-rate scheduler is working properly.
    * Every single scalar value in logs.

    Todo
    ----
    Upgrade this class to handle hyperparameter information and write different loss
    curves based on different values of a hyperparameter.
    """
    
    def __init__(self, repr_inputs: Union[tf.data.Dataset, Collection[tf.Tensor]],
                       log_dir: str = '.',
                       quiver_downsample: int = 32,
                       fig_args: dict = {},
                       quiver_args: dict = {}):
        """
        Parameters
        ----------
        repr_inputs : tf.data.Dataset or a collection of items
            Representative inputs to show for the image-based logging aspects of this class.
            These should be arranged according to the model's input specifications.
            Currently, that specification is: [moving_imgs, fixed_imgs, hparam, moving_lbls].
            THIS MAY CHANGE WITH THE NEW HYPERMORPH MODELS.
        model : tf.keras.Model or similar
            A model SUBSCRIBING TO THE OLD SYNTHMORPH MODEL FORMAT. This class currently does
            not work for the new Hypermorph models.
        log_dir : str
            Directory to which the log files should be written. Default is '.'.
        quiver_downsample : int
            The skip to use when sampling values of the deformation field for the quiver plot
            (essentially, the quiver plot is downsampled using `[::quiver_downsample]` before
            plotting for clarity). Default is 32.
        fig_args : dict
            Arguments to use when creating the Matplotlib figure for use in the quiver plot.
            Default is {}.
        quiver_args : dict
            Arguments to use when calling ax.quiver when making the quiver plot. Default is {}.
        """
        self.repr_inputs = repr_inputs
        self.fig_args = fig_args
        if 'figsize' not in self.fig_args.keys(): self.fig_args['figsize'] = (3,3) # default figsize to avoid issues with quiver
        self.quiver_args = quiver_args
        self.quiver_downsample = quiver_downsample
        self._fw = tf.summary.create_file_writer(log_dir)
        super().__init__()
    
    def on_epoch_end(self, epoch: int, logs: Collection[tf.Tensor] = None):
        """
        Logging function called at the end of every epoch.

        Parameters
        ----------
        epoch : int
            The epoch number.
        logs : dict or None
            The scalar logs to write to disk in addition to the standard values. Default is None.
        """

        # Get the deformation fields.
        repr_inputs = self.repr_inputs if not isinstance(self.repr_inputs, tf.data.Dataset) else next(iter(self.repr_inputs))        
        repr_outputs = self.model.predict(self.repr_inputs) if isinstance(self.repr_inputs, tf.data.Dataset) \
                                                            else self.model(self.repr_inputs)
        moving_imgs, fixed_imgs, _, moving_lbls = repr_inputs[0]
        _, _, fixed_lbls = repr_inputs[1]
        moved_imgs, deffs, moved_lbls = repr_outputs
        
        # Imgs and labels
        moving_imgs, moved_imgs, fixed_imgs = map(lambda img: tf.clip_by_value(img,
                                                                               clip_value_min=0,
                                                                               clip_value_max=1),
                                                  (moving_imgs, moved_imgs, fixed_imgs))
        moving_lbls, moved_lbls, fixed_lbls = map(lambda lbls: tf.expand_dims(normalize(tf.math.argmax(lbls, axis=-1),
                                                                                        min_value=0,
                                                                                        max_value=1,
                                                                                        lib=tf.experimental.numpy), axis=-1),
                                                  (moving_lbls, moved_lbls, fixed_lbls))
        
        # mags and angles
        mags = tf.expand_dims(tf.math.reduce_sum(deffs**2,
                                                 axis=-1)**0.5,
                              axis=-1)
        mags_norm = normalize(mags, min_value=0, max_value=1, lib=tf.experimental.numpy)
        angles = tf.expand_dims(tf.math.atan2(deffs[...,1],
                                              deffs[...,0]), axis=-1)
        angles_norm = normalize(angles, min_value=0, max_value=1, lib=tf.experimental.numpy)
        
        # deffs, with default size if not supplied (avoid errors with quiver)
        if 'figsize' not in self.fig_args.keys(): self.fig_args['figsize'] = (3,3)
        quiver_fn = lambda input_tuple: def2img(input_tuple[0],
                                                 downsample=self.quiver_downsample,
                                                 fig_args=self.fig_args,
                                                 quiver_args={**self.quiver_args,
                                                              'C': input_tuple[1]})
        deffs_imgs = tf.map_fn(quiver_fn,
                               (deffs, mags_norm),
                               dtype=tf.uint8,
                               fn_output_signature=None)
        
        # write logs
        with self._fw.as_default():
            tf.summary.image('moving_imgs', moving_imgs, step=epoch)
            tf.summary.image('fixed_imgs', fixed_imgs, step=epoch)
            tf.summary.image('moving_lbls', moving_lbls, step=epoch)
            tf.summary.image('fixed_lbls', fixed_lbls, step=epoch)
            tf.summary.image('moved_imgs', moved_imgs, step=epoch)
            tf.summary.image('moved_lbls', moved_lbls, step=epoch)
            tf.summary.image('mags', mags_norm, step=epoch)
            tf.summary.image('angles', angles_norm, step=epoch)
            tf.summary.image('deffs', deffs_imgs, step=epoch)
            tf.summary.scalar('mags_max', tf.math.reduce_max(mags), step=epoch)
            tf.summary.scalar('mags_min', tf.math.reduce_min(mags), step=epoch)
            tf.summary.scalar('angles_max', tf.math.reduce_max(angles), step=epoch)
            tf.summary.scalar('angles_min', tf.math.reduce_min(angles), step=epoch)
            tf.summary.scalar('lr', self.model.optimizer.lr, step=epoch)
            if logs is not None:
                for nm, val in logs.items():
                    tf.summary.scalar(name=nm, data=val, step=epoch)
