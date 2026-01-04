"""
UTILS.IMAGE.TISSUE

Functions for cleaning up tissue images.

Author: Peter Lais
Last updated: 10/15/2022
"""

from typing import Collection, Union
import numpy as np
from PIL import Image
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
from skimage.morphology import dilation, area_opening, erosion, remove_small_holes

from .image import resize_to_max_dim_pil

def posfreq(mask: np.ndarray, roi: Collection[int] = None) -> float:
    """
    Determine the percentage of hits in a patch of a mask.

    Parameters
    ----------
    mask : np.ndarray
        Two-dimensional mask.
    roi : length-4 list, optional
        Region of interest in the form start_r, start_c, end_r, end_c. If
        None, the whole mask is used.

    Returns
    -------
    float
        Frequency of positive occurrences in the region of interest.
    """
    # Basic assertions.
    assert roi.size == 4, "ROI must have four points."
    assert mask.ndim == 2, "Mask must be two-dimensional."

    # Determine the number of positive locations and the area.
    if roi is None:
        positive_locs = np.sum(mask) 
        area = np.prod(mask.shape)
    else:
        start_r, start_c, end_r, end_c = roi
        assert end_r >= start_r, "End row must be >= start row."
        assert end_c >= start_c, "End column must be >= start column."
        patch = mask[start_r:end_r, start_c:end_c]
        area = np.prod(patch.shape)
        positive_locs = np.sum(patch)
    
    # Return the frequency of positive locations over area.
    return positive_locs / area

def istissue(mask: np.ndarray, roi: Collection[int] = None,
             cutoff: float = 0.05) -> bool:
    """
    Determine if tissue exists based on a mask and a cutoff in a given ROI.

    Parameters
    ----------
    mask : np.ndarray
        Two-dimensional mask.
    roi : array with shape [..., 4], optional
        Collection of regions of interest with the last dimension taking
        the form form start_r, start_c, end_r, end_c. If None, the whole
        mask is used.
    cutoff : float
        Minimum frequency to indicate a mask/ROI contains tissue. Default
        is 0.05

    Returns
    -------
    np.ndarray
        Either a single boolean (if roi is None) or a collection of booleans
        with shape [...]. Both will be of type np.ndarray for simplicity.
    """
    assert cutoff >= 0 and cutoff <= 1, "Cutoff must fall within the range [0, 1]."

    if roi is None: result = np.asarray(posfreq(mask, roi))
    else: result = np.apply_along_axis(func1d=lambda roi, mask: posfreq(mask, roi),
                                       axis=-1,
                                       arr=roi,
                                       mask=mask)
    return result > cutoff


# img is numpy rgb
def tissue_mask_basic(img: np.ndarray, downsampled_max_dim: Union[int, None] = 512,
                      area_threshold: int = 512, connectivity: int = 4, rounds=1) -> np.ndarray:
    """
    A simple function that masks the regions of histology slides where no tissues are present.

    Parameters
    ----------
    img : np.ndarray
        Tissue image to mask.
    downsampled_max_dim : int
        The maximum dimension of the reduced-resolution tissue image on which masking functions
        will be applied.
    area_threshold : int
        The minimum tissue area in pixels for tissue to be kept in the tissue mask. Smaller numbers
        allow for more granular tissue masks.
    connectivity: int
        The connectivities to use during morphological operaitons. See sklearn.morphology
        for permitted values.

    Returns
    -------
    np.ndarray
        The tissue mask where True values indicate regions of tissue.
    """

    # Resize the image to thumbnail dimensions.
    if downsampled_max_dim is not None:
        pil_img = Image.fromarray(img)
        orig_size = pil_img.size
        downsample_factor_thumbnail = (np.amax(img.shape[:2]) / downsampled_max_dim)
        img = np.array(pil_img.resize([round(dim / downsample_factor_thumbnail)
            for dim in pil_img.size]))

    # Get the thresholded image based on hue
    img_hue = rgb2hsv(img)[..., 0]
    tissue_hue_threshold = threshold_otsu(img_hue)
    img_hue_mask = img_hue > tissue_hue_threshold

    # Do some operations that generally lead to a good mask
    img_hue_mask = dilation(area_opening(erosion(area_opening(
        img_hue_mask, connectivity=connectivity, area_threshold=area_threshold)),
        connectivity=connectivity, area_threshold=area_threshold))
    if rounds>1:
        counter=1
        for counter in range(2,rounds):
            img_hue_mask=dilation(area_opening(img_hue_mask))
        img_hue_mask=remove_small_holes(img_hue_mask, area_threshold=1000)
        
    if downsampled_max_dim is not None:
        return np.array(Image.fromarray(img_hue_mask).resize(orig_size,
                                                             resample=Image.Resampling.NEAREST)) #Resampling
    else: return img_hue_mask

# V2

def standard_mask_fn(pil_im: Image, max_mask_dim: int = 512,connectivity=4,rounds=1) -> Image:
    """
    Standard mask function used by this library to mask histology slides. See tissue_mask_basic
    for implementation details.

    Parameters
    ----------
    pil_im : PIL Image
        PIL image to mask.
    max_mask_dim : int
        The maximum dimension of the reduced-resolution tissue image on which masking functions
        will be applied.
    
    Returns
    -------
    PIL Image
        The tissue mask where True values indicate regions of tissue.
    """
    thumbnail = resize_to_max_dim_pil(pil_im, max_mask_dim)
    thumbnail_mask = tissue_mask_basic(img=np.asarray(thumbnail),
                                       downsampled_max_dim=max_mask_dim,
                                       area_threshold=max_mask_dim,
                                       connectivity=connectivity,rounds=rounds)
    thumbnail_mask_pil = Image.fromarray(thumbnail_mask)
    return thumbnail_mask_pil.resize(pil_im.size, Image.Resampling.NEAREST).convert('1')