"""
LOGGING

Utility functions to help with logging.

Author: Peter Lais
Last updated: 10/15/2022
"""

import logging
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import warnings
from contextlib import contextmanager
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from pathlib import Path
from tensorflow import keras
from .utils import minmax
from ..metrics import ncc_metric

# Ignore logging in some contexts
@contextmanager
def ignore_nonfatal_tf_logs():
    """
    Ignore any non-error level logs from Tensorflow for any code executed within this context.
    """
    logger = tf.get_logger()
    source_level = logger.level
    logger.setLevel(logging.ERROR)
    yield
    logger.setLevel(source_level)

class RepresenativeIO(keras.callbacks.Callback):
    """
    Print out representative outputs from a model.
    """

    def __init__(
        self,
        representative_inputs,
        log_dir=".",
        consider_out="all",  # 'all' or list of numbers or none
        consider_out_once=False,
        consider_in=None,  # 'all' or list of numbers or none
        consider_in_once=False,
        log_y_true=True,
        name="representative_io",
        use_base_model=False,
    ):
        super().__init__()

        if isinstance(log_dir, Path):
            log_dir = log_dir.as_posix()

        self._predict = isinstance(representative_inputs, tf.data.Dataset)
        self.representative_inputs = representative_inputs
        self.file_writer = tf.summary.create_file_writer(log_dir)
        self.name = name

        if isinstance(consider_out, int):
            consider_out = [consider_out]
        self.consider_out = consider_out

        if isinstance(consider_in, int):
            consider_in = [consider_in]
        self.consider_in = consider_in

        self.consider_out_once = consider_out_once
        self.consider_in_once = consider_in_once

        self._fresh = True
        self.log_y_true = log_y_true
        self.use_base_model = use_base_model

    def on_epoch_end(self, epoch, logs=None):

        model = (
            self.model if not self.use_base_model else self.model.references.base_model
        )
        outputs = (
            model.predict(self.representative_inputs)
            if self._predict
            else model(self.representative_inputs)
        )

        # Write the relevant outputs.
        if (
            not self.consider_out_once or (self.consider_out_once and self._fresh)
        ) and self.consider_out is not None:

            # If no names provided, make custom ones.
            if not isinstance(outputs, dict):
                outputs = {"out%d" % i: output for i, output in enumerate(outputs)}

            for i, (output_name, output) in enumerate(outputs.items()):
                if self.consider_out == "all" or i in self.consider_out:
                    if output.shape[-1] not in (1, 3):
                        output = tf.math.argmax(output, axis=-1)
                        output = tf.expand_dims(output, axis=-1)
                        output = (output - tf.math.reduce_min(output)) / (
                            tf.math.reduce_max(output) - tf.math.reduce_min(output)
                        )
                    with self.file_writer.as_default():
                        tf.summary.image(
                            name="%s_%s" % (self.name, output_name),
                            data=output,
                            step=epoch,
                        )

        # Write the relevant inputs.
        # These shouldn't be in a dictionary format since we pass in the inputs
        # ourselves (as a Dataset) rather than get it from the model.
        def handle_input(input, i, override=False):
            if self.consider_in == "all" or i in self.consider_in or override:
                if input.shape[-1] not in (1, 3):
                    input = tf.math.argmax(input, axis=-1)
                    input = tf.expand_dims(input, axis=-1)
                    input = (input - tf.math.reduce_min(input)) / (
                        tf.math.reduce_max(input) - tf.math.reduce_min(input)
                    )
                with self.file_writer.as_default():
                    tf.summary.image(
                        name="%s_in%s" % (self.name, i), data=input, step=epoch
                    )

        if (
            not self.consider_in_once or (self.consider_in_once and self._fresh)
        ) and self.consider_in is not None:
            for i, input in enumerate(self.representative_inputs):
                # Get only the inputs, not the expected target.
                if self._predict:
                    for j, inputt in enumerate(input[0]):
                        handle_input(inputt, j)
                    if self.log_y_true:
                        handle_input(input[1], j + 1, override=True)
                else:
                    handle_input(input, i)

        self._fresh = False


class UNetDiagnosticIO(keras.callbacks.Callback):
    """
    A callback to write images to Tensorboard.
    BUG: not all tensors used here, elicits warning from tensorflow
    """

    def __init__(
        self,
        representative_inputs,
        moved_img_output=0,
        fixed_img_input=1,
        def_output=1,
        fig_dpi=300,
        quiver_axis_resolution=32,  # approximately the number of arrows on each axis of quiver
        log_dir=".",
        name="def_output_visualizer",  # name is not recorded with diagnostics anymore
        use_base_model=False,
        use_ncc=False,
        visualize_differences=False,
    ):
        super().__init__()

        if isinstance(log_dir, Path):
            log_dir = log_dir.as_posix()

        self._predict = isinstance(representative_inputs, tf.data.Dataset)
        self.representative_inputs = representative_inputs
        self.file_writer = tf.summary.create_file_writer(log_dir)
        self.def_output = def_output
        self.name = name
        self.dpi = fig_dpi
        self.quiver_axis_resolution = quiver_axis_resolution
        self.use_base_model = use_base_model
        self.use_ncc = use_ncc
        self.moved_img_output = moved_img_output
        self.fixed_img_input = fixed_img_input
        self.visualize_differences = visualize_differences

    def on_epoch_end(self, epoch, logs=None):

        # Get the deformation fields.
        model = (
            self.model if not self.use_base_model else self.model.references.base_model
        )
        outputs = (
            model.predict(self.representative_inputs)
            if self._predict
            else model(self.representative_inputs)
        )

        # Determine what operations to perform.
        # TO DO: assert that moved_fixed_defined is true if any of ncc_active, diffs_active are true
        moved_fixed_defined = (
            self.moved_img_output is not None and self.fixed_img_input is not None
        )
        ncc_active = self.use_ncc and moved_fixed_defined
        diffs_active = self.visualize_differences and moved_fixed_defined

        # Make an average ncc.
        if ncc_active:
            moved_img = outputs[self.moved_img_output]
            curr_moved_img_index = 0
            curr_ncc = 0
            for i, batch in enumerate(self.representative_inputs):
                inputs = batch[0]
                fixed_img_batch = inputs[self.fixed_img_input]
                moved_img_batch = moved_img[
                    curr_moved_img_index : curr_moved_img_index + len(fixed_img_batch)
                ]
                curr_ncc += tf.math.reduce_mean(
                    ncc_metric(
                        tf.convert_to_tensor(fixed_img_batch),
                        tf.convert_to_tensor(moved_img_batch),
                    )
                )
                curr_moved_img_index += len(fixed_img_batch)
            curr_ncc /= i + 1

        # This is def_output, not us. Rename.
        if self.def_output is not None:
            def_output = outputs[self.def_output]

            # Mark unused outputs as used.
            # for i, output in enumerate(outputs):
            #  if i != self.us_output:
            #    output.mark_used()

            mags = tf.expand_dims(tf.math.reduce_sum(def_output ** 2, axis=-1), axis=-1)
            angles = tf.expand_dims(
                tf.math.atan2(def_output[..., 1], def_output[..., 0]), axis=-1
            )

            # See if this works at all
            # TODO: Plot quiver plots for the whole batch (need to do one at a time)
            # and eliminate the magnitude/angle plots. Can keep the biggest/least mag
            # (and possibly angle) over time though.

            def def_output_to_quiver(
                def_output_single, C=None, fig_args={}, quiver_args={}
            ):
                # Generates a rasterized figure from a def_output.
                # Not a tf.function since it involves numpy arrays and might not play
                # nice with a graph.

                # Make the figure and the quiver
                # Invert the y-axis since image coordinates start from top-left
                # Tight layout to avoid margins in tensorboard
                fig = Figure(**fig_args)
                canvas = FigureCanvas(fig)
                ax = fig.gca()
                X, Y = np.indices(def_output_single.shape[:-1])
                U, V = def_output_single[..., 0], def_output_single[..., 1]

                # Get the resolution to sample by
                if self.quiver_axis_resolution is not None:
                    sample_by_x = (
                        def_output_single.shape[0] // self.quiver_axis_resolution
                    )
                    sample_by_y = (
                        def_output_single.shape[1] // self.quiver_axis_resolution
                    )
                    if not sample_by_x:
                        warnings.warn(
                            "quiver_axis_resolution exceeds maximum resolution"
                            " in x-direction, setting spacing to 1. Try setting quiver_axis_resolution"
                            " to None."
                        )
                    if not sample_by_y:
                        warnings.warn(
                            "quiver_axis_resolution exceeds maximum resolution"
                            " in y-direction, setting spacing to 1. Try setting quiver_axis_resolution"
                            " to None."
                        )
                else:
                    sample_by_x, sample_by_y = 1, 1

                # Make a quiver plot, for some reason ax.axis('square') eliminates all data
                ax.quiver(
                    X[::sample_by_x, ::sample_by_y],
                    Y[::sample_by_x, ::sample_by_y],
                    U[::sample_by_x, ::sample_by_y],
                    V[::sample_by_x, ::sample_by_y],
                    C[::sample_by_x, ::sample_by_y],
                    **quiver_args
                )
                ax.invert_yaxis()
                ax.axis("equal")
                plt.tight_layout()

                # Rasterize the figure
                canvas.draw()
                width, height = map(int, fig.get_size_inches() * fig.get_dpi())
                return tf.convert_to_tensor(
                    np.frombuffer(canvas.tostring_rgb(), dtype="uint8").reshape(
                        height, width, 3
                    ),
                    dtype=tf.uint8,
                )

            quiver_fn = lambda input_tuple: def_output_to_quiver(
                input_tuple[0],
                C=input_tuple[1],
                fig_args={"dpi": self.dpi},
                quiver_args={},
            )
            def_output_imgs = tf.map_fn(
                quiver_fn, (def_output, mags), dtype=tf.uint8, fn_output_signature=None
            )

            with self.file_writer.as_default():
                # Record everything.
                tf.summary.image("mags", minmax(mags), step=epoch)
                tf.summary.image("angles", minmax(angles), step=epoch)
                tf.summary.scalar("mags_max", tf.reduce_max(mags), step=epoch)
                tf.summary.scalar("mags_min", tf.reduce_min(mags), step=epoch)
                tf.summary.scalar("angles_max", tf.reduce_max(angles), step=epoch)
                tf.summary.scalar("angles_min", tf.reduce_min(angles), step=epoch)
                # tf.summary.scalar('loss', logs['loss'], step=epoch)
                tf.summary.scalar("lr", self.model.optimizer.lr, step=epoch)
                tf.summary.image("def_output_quiver", def_output_imgs, step=epoch)

        with self.file_writer.as_default():
            # Record all metrics and losses
            if ncc_active:
                tf.summary.scalar("ncc", curr_ncc, step=epoch)

            if diffs_active:
                all_fixed_imgs = tf.concat(
                    values=[
                        batch[0][self.fixed_img_input]
                        for batch in self.representative_inputs
                    ],
                    axis=0,
                )
                moved_imgs = outputs[self.moved_img_output]
                diffs = tf.expand_dims(
                    tf.math.abs(
                        tf.math.reduce_mean(moved_imgs, axis=-1)
                        - tf.math.reduce_mean(all_fixed_imgs, axis=-1)
                    ),
                    axis=-1,
                )
                tf.summary.image("moving_fixed_diffs_mag", diffs, step=epoch)

            tf.summary.scalar("lr", self.model.optimizer.lr, step=epoch)
            for key, val in logs.items():
                tf.summary.scalar(name=key, data=val, step=epoch)