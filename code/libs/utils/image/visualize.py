"""
UTILS.IMAGE.VISUALIZE

Visualization functions. As the only function in this library is for use with Hypermorph
models (which are deprecated), this module may be removed eventually.

Author: Peter Lais
Last updated: 10/15/2022
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import imageio
import os
from ...losses import dice_loss, grad_loss
from .convert import plt2img

def visualize_results_hypermorph(model,
                                 dataset_batch,
                                 hyperparameters,
                                 out_dir='.',
                                 format='.mp4',
                                 results_for=None,
                                 dpi=None,
                                 fps=10):
    """
    Makes a GIF summarizing HyperMorph network performance.

    This function is outdated. If it is not used elsewhere, it will be deleted.
    I am therefore not documenting it.
    """

    input_batch, target_batch, _ = dataset_batch
    moving_imgs, fixed_imgs, _, moving_lbls = input_batch
    fixed_imgs, _, fixed_lbls = target_batch

    # preparation
    if '__len__' not in dir(hyperparameters): hyperparameters = [hyperparameters]
    hparams = sorted(hyperparameters)
    if results_for is not None and '__len__' not in dir(results_for):
        results_for = [results_for]

    # run the code on each output
    # https://jakevdp.github.io/PythonDataScienceHandbook/04.08-multiple-subplots.html
    hparam_data = {}
    for hparam in hparams:

        # Get model outputs
        temp_input_batch = [moving_imgs, fixed_imgs, tf.ones((len(moving_imgs), 1)) * hparam, moving_lbls]
        output_batch = model(temp_input_batch)
        moved_imgs, deffs, moved_lbls = output_batch

        # Store results
        # Dice loss is -(dice)
        dice_orig = [float(-dice_loss(tf.expand_dims(f, axis=0), tf.expand_dims(m, axis=0)))
            for f, m  in zip(fixed_lbls, moving_lbls)]
        dice_reg = [float(-dice_loss(tf.expand_dims(f, axis=0), tf.expand_dims(m, axis=0)))
            for f, m  in zip(fixed_lbls, moved_lbls)]
        grad = [float(grad_loss(None, tf.expand_dims(d, axis=0))) for d in deffs]
        hparam_data[hparam] = {'moving_img': moving_imgs,
                               'moved_img': moved_imgs,
                               'fixed_img': fixed_imgs,
                               'moving_lbl': tf.math.argmax(moving_lbls, axis=-1),
                               'moved_lbl': tf.math.argmax(moved_lbls, axis=-1),
                               'fixed_lbl': tf.math.argmax(fixed_lbls, axis=-1),
                               'dice_orig': dice_orig,
                               'dice_reg': dice_reg,
                               'grad': grad}
        
    # Make the graphs
    for i in range(len(input_batch[0])) if results_for is None else results_for:
        # Get the structures
        frames = []
        for j, hparam in enumerate(hparams):
            # Make the figure layout
            fig, ((moving_img_fig, moved_img_fig, fixed_img_fig, dummy, performance_chart),
                (moving_lbl_fig, moved_lbl_fig, fixed_lbl_fig, dummy2, grad_chart)) = plt.subplots(2, 5, figsize=(8,3), dpi=dpi, sharex='col', gridspec_kw={'width_ratios': [1, 1, 1, 0, 3]})
            moving_img_fig.set_title('Moving image')
            moving_lbl_fig.set_title('Moving label')
            moved_img_fig.set_title('Moved image')
            moved_lbl_fig.set_title('Moved label')
            fixed_img_fig.set_title('Fixed image')
            fixed_lbl_fig.set_title('Fixed label')
            moving_img_fig.axis('off')
            moving_lbl_fig.axis('off')
            moved_img_fig.axis('off')
            moved_lbl_fig.axis('off')
            fixed_img_fig.axis('off')
            fixed_lbl_fig.axis('off')
            dummy.axis('off')
            dummy2.axis('off')
            performance_chart.set_xscale('log')
            performance_chart.set_xlabel('Reg. param')
            performance_chart.set_ylabel('Dice score')
            grad_chart.set_xlabel('Reg. param')
            grad_chart.set_ylabel('|Grad|')

            dice_origs = [val['dice_orig'][i] for val in hparam_data.values()]
            dice_regs = [val['dice_reg'][i] for val in hparam_data.values()]
            grads = [val['grad'][i] for val in hparam_data.values()]

            moving_img_fig.imshow(hparam_data[hparam]['moving_img'][i])
            moved_img_fig.imshow(hparam_data[hparam]['moved_img'][i])
            fixed_img_fig.imshow(hparam_data[hparam]['fixed_img'][i])
            moving_lbl_fig.imshow(hparam_data[hparam]['moving_lbl'][i])
            moved_lbl_fig.imshow(hparam_data[hparam]['moved_lbl'][i])
            fixed_lbl_fig.imshow(hparam_data[hparam]['fixed_lbl'][i])
            performance_chart.plot(hparams, dice_regs)
            grad_chart.plot(hparams, grads)
            performance_chart.plot(hparam, dice_regs[j], 'ko')
            grad_chart.plot(hparam, grads[j], 'ko')
            fig.tight_layout()

            # Rasterize image
            frames.append(plt2img(fig))
            plt.close(fig)
        
        # Save
        writer = imageio.get_writer(os.path.join(out_dir, 'hypermorph_viz_%d%s' % (i, format)), fps=fps)
        for im in frames:
            writer.append_data(im)
        writer.close()