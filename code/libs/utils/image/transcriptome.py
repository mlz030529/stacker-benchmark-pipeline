"""
UTILS.IMAGE.TISSUE

Functions for cleaning up tissue images.

Author: Yu Bai
Last updated: 11/07/2022
"""




from scipy import ndimage
from scipy.ndimage.filters import gaussian_filter
import cv2
from scipy.interpolate import splprep, splev
import math
from PIL import ImageFilter
import scipy
import numpy as np
import pandas as pd

def getCountourImage2(seg_image):
    filters = []
    for i in [0, 1, 2]:
     for j in [0, 1, 2]:
        filter = np.zeros([3,3], dtype=int)
        if i ==1 and j==1:
            pass
        else:
            filter[i][j] = -1
            filter[1][1] = 1
            filters.append(filter)

    convolved_images = []
    for filter in filters:
        convoled_image = ndimage.correlate(seg_image, filter, mode='reflect')
        convolved_images.append(convoled_image)
    convoled_images = np.add.reduce(convolved_images)
    seg_image = np.where(convoled_images != 0, 255, 0)
    return seg_image


def makeContourImg1(adata,lbl_map,maskval=0,med_filter_size=4,trimMax=3,res="hires",tomerge=None):
    spatial_id=list(adata.uns.keys())[0]
    library_id=list(adata.uns[spatial_id].keys())[0]

    scale_factor=adata.uns[spatial_id][library_id]['scalefactors']['tissue_'+res+'_scalef']
    rowMin=adata.obsm['spatial'][:,1].min()*scale_factor
    rowMax=adata.obsm['spatial'][:,1].max()*scale_factor
    colMin=adata.obsm['spatial'][:,0].min()*scale_factor
    colMax=adata.obsm['spatial'][:,0].max()*scale_factor

    rowDelta=(rowMax-rowMin)*1.0/math.sqrt(len(adata.obsm['spatial'][:,1]))
    colDelta=(colMax-colMin)*1.0/math.sqrt(len(adata.obsm['spatial'][:,0]))
    
    test=lbl_map.copy()
    test[0:math.floor(max(1,rowMin-rowDelta)),:]=0
    test[math.ceil(min(lbl_map.shape[0]-1,rowMax+rowDelta)):-1,:]=0
    test[:,0:math.floor(max(1,colMin-colDelta))]=0
    test[:,math.ceil(min(lbl_map.shape[1]-1,colMax+colDelta)):-1]=0
    cp_warped=test.copy()
    if tomerge is not None:
     for i in range(len(tomerge)):
        ref=tomerge[i][0]
        for j in tomerge[i]:
            test[np.where(test==j)]=ref
     lbl_map_mini_warped=test

    contour2=getCountourImage2(lbl_map)
    contourimg2=np.zeros((lbl_map.shape[0],lbl_map.shape[1],3))
    contourimg2[:,:,0]=contour2
    contourimg2[:,:,1]=contour2
    contourimg2[:,:,2]=contour2
    contourimg2=scipy.ndimage.median_filter((contourimg2).astype(np.uint8), size=med_filter_size)
    contourimg2=contourimg2/255
    
    tissueEdge=np.where((cp_warped==maskval)&(contourimg2[:,:,0]>0))

    arm0=np.array(tissueEdge[0])
    arm1=np.array(tissueEdge[1])
    for i in range(tissueEdge[0].shape[0]):
     for s in range(-trimMax,trimMax+1):
        for t in range(-trimMax,trimMax+1):
            if tissueEdge[0][i]+s>=0 and tissueEdge[0][i]+s<lbl_map.shape[0] and \
                tissueEdge[1][i]+t>=0 and tissueEdge[1][i]+t<lbl_map.shape[1]:
                 arm0=np.append(arm0,tissueEdge[0][i]+s)
                 arm1=np.append(arm1,tissueEdge[1][i]+t)

    tissueEdge_add=(arm0,arm1)
    contourimg2[tissueEdge_add]=0
    return lbl_map,contourimg2



## generate label map based on known cluster assignment of tissue spots or cells
## follow the voronoi rule to assign label to each pixel
def dilateLBL(annots,stobj,resolution="cluster",offset=100,res="hires"):
    """
    annots: annotation table for spots or cells, must contains at least 2 columns: 
            - spotID (id of spots/cells)
            - cluster (cluster assignment of spots/cells, numerical values)
    stobj:  AnnData object representing the input spatial transcriptome slice 
    resolution: the column name used for the cluster information (default='cluster') in the annotation table
    offset: a constant value added to the numerical cluster labels, making easier visualization downstream
    res:  key value to access the tissue image stored in stobj 
    
    """
    
    spot_coords=pd.DataFrame(stobj.obsm['spatial'].copy(),index=stobj.obs.index)
    spot_coords=spot_coords.loc[list(annots['spotID'])]
    #print("number of "+resolution+"="+str(annots[resolution].unique().size))
    spatial_id=list(stobj.uns.keys())[0]
    library_id=list(stobj.uns['spatial'].keys())[0]
    maxH=stobj.uns[spatial_id][library_id]['images'][res].shape[0]
    maxW=stobj.uns[spatial_id][library_id]['images'][res].shape[1]
    scale_factor=stobj.uns[spatial_id][library_id]['scalefactors']['tissue_'+res+'_scalef']
    scaled_spot_coords=spot_coords.to_numpy()*scale_factor

    lbl_map=np.zeros((maxW,maxH))
    for y in range(maxH):
        for x in range(maxW):
            new_point=np.array([[x, y]])
            # find nearest spot and break tie randomly
            dist=np.sum((scaled_spot_coords - new_point)**2, axis=1)
            whichSpot=np.random.choice(np.where(dist == dist.min())[0])
            lbl_map[x,y]=annots.iloc[whichSpot][resolution]+offset
        
    lbl_map[np.transpose(stobj.uns[spatial_id][library_id]['images'][res][:,:,0])==0]=0
    return lbl_map

def createCompositeImg(adata,clust_file,res="hires",maskval=0,med_filter_size=4,trimMax=2):
    """
    adata:  AnnData object representing the input spatial transcriptome slice 
    clust_file: name of the annotation file, must contains at least 2 columns: 
            - spotID (id of spots/cells)
            - cluster (cluster assignment of spots/cells, numerical values)
    res:  key value to access the tissue image stored in adata
    maskval: parameters passed onto makeContourImg1
    med_filter_size: paramter passed onto scipy.ndimage.median_filter
    trimMax: parameters passed onto makeContourImg1
    """
    spatial_id=list(adata.uns.keys())[0]
    library_id=list(adata.uns[spatial_id].keys())[0]
    annots=pd.read_csv(clust_file)
    annots['spotID']=annots['spotID'].replace('^sp1_', '', regex=True, inplace=False)
    annots.index=annots['spotID']
    ol=list(set(annots.index.tolist()) & set(adata.obs.index.tolist()))
    annots=annots.loc[ol]
    adata=adata[annots.index,]
    
    lbl_map=np.transpose(dilateLBL(annots,adata,resolution='cluster',offset=100))
    [lbl_map,contourimg]=makeContourImg1(adata,lbl_map,maskval=maskval,med_filter_size=med_filter_size,trimMax=trimMax)
    
    orgimg=adata.uns[spatial_id][library_id]['images'][res].copy()
    combo_unwarped=orgimg
    #mixing proportion of the contour image vs. the tissue image = 0.3:0.7
    combo_unwarped[np.where(contourimg>0)]=0.7*orgimg[np.where(contourimg>0)]+0.3*1 
    adata.uns[spatial_id][library_id]['images'][res]=combo_unwarped

    return adata