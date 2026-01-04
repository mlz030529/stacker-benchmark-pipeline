"""
UTILS.IO.SCANPY

Functions for making loading and saving of scanpy data easier.

Author: Peter Lais
Last updated: 10/15/2022
"""

import scanpy as sc
import numpy as np
import pandas as pd
from PIL import Image

def adata_load_fun(series, spatial_target='hires'):
    """
    Utility function to load AnnData objects and relevant parameters into a row
    of a Pandas DataFrame (essentially a Series). This may be used on DataFrames
    containing series with the information specified in the **Parameters** section.

    Exclusively used for reading 10X Visium data.

    Parameters
    ----------
    series : pd.Series
        A series containing the following values (see `scanpy.read_visium` for more
        information):

        * `path`: path to the folder containing the .hd5 counts file.
        * `genome`: name of the genome to include, if relevant.
        * `count_file`: name for the count file if deviating from the default naming schema.
        * `library_id`: library id to include, if relevant.
        * `source_image_path`: path to the source image used with spatial data.

    spatial_target : str
        Spatial information to use when populating table. Default is 'hires'; see AnnData.uns
        for more information about what resolutions are available.
    
    Returns
    -------
    pd.Series
        Appropriately populated pd.Series containing:

        * adata-original: original AnnData file output by `scanpy.read_visium`.
        * scalefactor: scale factor associated with spatial_target.
        * spatial-scaled: original AnnData file output by `scanpy.read_visium`.
        * width: width of the image associated with spatial_target.
        * height: height of the image associated with spatial_target.
        * image: image associated with spatial_target.
    """
    adata = sc.read_visium(path=series['path'],
                           genome=series['genome'],
                           count_file=series['count_file'],
                           library_id=series['library_id'],
                           source_image_path=series['source_image_path'])
    spatial_container = next(iter(adata.uns['spatial'].values()))
    image = Image.fromarray((spatial_container['images'][spatial_target] * 255).astype(np.uint8)).convert('RGB')
    scalefactor = spatial_container['scalefactors']['tissue_%s_scalef' % spatial_target]
    width, height = np.array(image.size)
    spatial_scaled = adata.obsm['spatial'] * scalefactor
    return pd.Series([adata, scalefactor, spatial_scaled, width, height, image],
                     index=['adata-original', 'scalefactor', 'spatial-scaled', 'width', 'height', 'image'])
