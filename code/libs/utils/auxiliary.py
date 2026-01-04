# auxiliary functions 
# cropAdata, padAdata, scaleData, iterModel, scale_spatial_coords2, makeContourImg, getCountourImage2
import math
from PIL import Image
from keras import backend as K
import numpy as np
import pandas as pd
import anndata as ad
from itertools import product
import voxelmorph as vxm
import tensorflow as tf
import re
import ants
import SimpleITK as sitk
from libs.utils.image.tissue import standard_mask_fn
from libs.alignment import sc_register, ants_align_pil,ants_align_pil2

from libs.utils.image import (pad_img as pad_img_to_even_tile,
                          unpad_img as unpad_img_from_even_tile,
                          stitch,
                          resize_to_max_dim_pil)
from libs.utils.transform import (scale_affine,
                              ants2itk_affine,
                              apply_itk_trf_image,
                              channeltransform,
                              register_spatial_with_itk_points,
                              register_spatial_with_itk_raster,
                              register_spatial_with_def_field_points,
                              register_spatial_with_def_field_raster)


from scipy import ndimage
from scipy.ndimage.filters import gaussian_filter
import cv2
from scipy.interpolate import splprep, splev


def scale_spatial_coords2(X, ref_min=None, ref_max=None, max_val=10.0):
    if ref_min is None:
        ref_min = X.min(0)
        #print("ref_min = "+str(ref_min))
    X = X - ref_min
    if ref_max is None:
        ref_max = X.max(0)
        #print("ref_max = "+str(ref_max))
    X = X / ref_max
    return X * max_val


def cropAdata(adata,spatial_key,library_id, padratio=0.1,addOnRatioMinX=0,addOnRatioMaxX=0,addOnRatioMinY=0,addOnRatioMaxY=0,doMask=False,withOuts=False,dev4search=0.2,ninterval4search=50,connectivity=4,rounds=1):
    adata_trim = adata.copy()
    newobsm=adata_trim.obsm.copy() # spot coords at full TIFF resolution
    newobs=adata_trim.obs.copy()
    newobsm[spatial_key]=newobsm[spatial_key][list(np.where(newobs['in_tissue'].to_numpy()==1)[0])] # new obsm['spatial']
    newobs=newobs.loc[newobs['in_tissue']==1] #new obs
    scalelow=adata_trim.uns[spatial_key][library_id]['scalefactors']['tissue_lowres_scalef']
    scalehigh=adata_trim.uns[spatial_key][library_id]['scalefactors']['tissue_hires_scalef']
    scale_imgIntensity=adata_trim.uns[spatial_key][library_id]['images']['hires']
    htif,wtif,ncolor=scale_imgIntensity.shape
    
    xy_indices=newobsm[spatial_key].copy()
    intissueMinx=(xy_indices[:,0].min())
    intissueMaxx=(xy_indices[:,0].max())
    intissueMiny=(xy_indices[:,1].min())
    intissueMaxy=(xy_indices[:,1].max())

    #@@ change to use tissue boundary not just in_tissue spot boundary to decide padding
    deltax=xy_indices[:,0].max()-xy_indices[:,0].min()
    deltay=xy_indices[:,1].max()-xy_indices[:,1].min()
#     deltax=(xy_indices[:,0].max()-xy_indices[:,0].min())*(1+addOnRatioX)
#     deltay=(xy_indices[:,1].max()-xy_indices[:,1].min())*(1+addOnRatioY)

    tissuelength=max(deltax,deltay)
    #print(tissuelength)
    
    bestratioX, bestratioY=optPadlength3(tissuelength=tissuelength, padratio=padratio,spotCoordsTIF=xy_indices, 
                                         scale_imgIntensity=scale_imgIntensity,
                                        scalefactor=scalehigh, dev4search=dev4search, ninterval4search=ninterval4search)
    print((bestratioX, bestratioY))
    bestpadlengthX=min(min(round(tissuelength*bestratioX),xy_indices[:,0].min()), wtif/scalehigh-xy_indices[:,0].max())#padding length
    bestpadlengthY=min(min(round(tissuelength*bestratioY),xy_indices[:,1].min()), htif/scalehigh-xy_indices[:,1].max()) #padding length
    #print([wtif,htif,scalehigh])
    #print([xy_indices[:,0].min(),xy_indices[:,0].max(),xy_indices[:,1].min(),xy_indices[:,1].max()])
    
    #print([tissuelength*bestratioX,xy_indices[:,0].min(), wtif/scalehigh-xy_indices[:,0].max()])
    #print([tissuelength*bestratioY,xy_indices[:,1].min(), htif/scalehigh-xy_indices[:,1].max()])
    #print([bestpadlengthX,bestpadlengthY])
    
#    padlength=min(round(tissuelength*padratio),min(xy_indices[:,0].min(),xy_indices[:,1].min())) #padding length

    # get the new boundary of the cropped image
    bestpadlengthX=bestpadlengthX*(1+addOnRatioMinX)
    bestpadlengthY=bestpadlengthY*(1+addOnRatioMinY)
    newMinX=xy_indices[:,0].min()-bestpadlengthX # col? 
    newMaxX=xy_indices[:,0].max()+bestpadlengthX
    newMinY=xy_indices[:,1].min()-bestpadlengthY # row?
    newMaxY=xy_indices[:,1].max()+bestpadlengthY

    
    #shift spot coords to match newly cropped img
    newobsm[spatial_key][:,0]=newobsm[spatial_key][:,0]-newMinX # = padlength
    newobsm[spatial_key][:,1]=newobsm[spatial_key][:,1]-newMinY


    hnewMinX=round(newMinX*scalehigh)+0 #PIL Image cropping is 0-index and end-exclusive
    hnewMaxX=round(newMaxX*scalehigh)+1
    hnewMinY=round(newMinY*scalehigh)+0
    hnewMaxY=round(newMaxY*scalehigh)+1


    lnewMinX=round(newMinX*scalelow)+0 #col/width
    lnewMaxX=round(newMaxX*scalelow)+1
    lnewMinY=round(newMinY*scalelow)+0 #row/height
    lnewMaxY=round(newMaxY*scalelow)+1


    #left, top, right, bottom
    if doMask:
        masked_lowres=maskNonTissue(adata_trim.uns[spatial_key][library_id]['images']['lowres'],intissueMinx,intissueMiny,intissueMaxx,intissueMaxy,scaling=None,connectivity=connectivity,rounds=rounds)
        masked_hires=maskNonTissue(adata_trim.uns[spatial_key][library_id]['images']['hires'],intissueMinx,intissueMiny,intissueMaxx,intissueMaxy,scaling=None,connectivity=connectivity,rounds=rounds)
        lcropped=Image.fromarray((masked_lowres * 255).astype(np.uint8)).convert('RGB').crop((lnewMinX,lnewMinY,lnewMaxX,lnewMaxY))
        hcropped=Image.fromarray((masked_hires* 255).astype(np.uint8)).convert('RGB').crop((hnewMinX,hnewMinY,hnewMaxX,hnewMaxY))
    else:
        lcropped=Image.fromarray((adata_trim.uns[spatial_key][library_id]['images']['lowres'] * 255).astype(np.uint8)).convert('RGB').crop((lnewMinX,lnewMinY,lnewMaxX,lnewMaxY))
        hcropped=Image.fromarray((adata_trim.uns[spatial_key][library_id]['images']['hires']* 255).astype(np.uint8)).convert('RGB').crop((hnewMinX,hnewMinY,hnewMaxX,hnewMaxY))

    adata_trim.uns[spatial_key][library_id]['images']['lowres']=(np.asarray(lcropped)/255).astype(np.float32)
    adata_trim.uns[spatial_key][library_id]['images']['hires']=(np.asarray(hcropped)/255).astype(np.float32)
    adata_trim.obs=newobs
    adata_trim.obsm=newobsm
    if withOuts:
        return [adata_trim, {'padlengthX':bestpadlengthX,'padlengthY':bestpadlengthY,'scalehigh':scalehigh,'tissuelength':tissuelength,
                             'hnewMinX':hnewMinX,'hnewMaxX':hnewMaxX,'hnewMinY':hnewMinY,'hnewMaxY':hnewMaxY,
                             'newMinX':newMinX,'newMaxX':newMaxX,'newMinY':newMinY,'newMaxY':newMaxY}]
    else:
        return adata_trim

def optPadlength(tissuelength, padratio, spotCoordsTIF,scale_imgIntensity, scalefactor, dev4search=0.2, ninterval4search=50):
    spotMinXtif=spotCoordsTIF[:,0].min()
    spotMinYtif=spotCoordsTIF[:,1].min()
    htif,wtif,ncolor=scale_imgIntensity.shape
    bestratioX=padratio
    #print((spotMinXtif,spotMinYtif))
    bestpadlengthX=min(min(round(tissuelength*bestratioX),spotMinXtif), wtif/scalefactor-spotCoordsTIF[:,0].max() ) #padding length
    bestdeltaX=abs((round(spotMinXtif*scalefactor)-round((spotMinXtif-bestpadlengthX)*scalefactor))-round(bestpadlengthX*scalefactor))

    bestratioY=padratio
    bestpadlengthY=min(min(round(tissuelength*bestratioY),spotMinYtif), htif/scalefactor-spotCoordsTIF[:,1].max()) #padding length
    bestdeltaY=abs((round(spotMinYtif*scalefactor)-round((spotMinYtif-bestpadlengthY)*scalefactor))-round(bestpadlengthY*scalefactor))
    for each in np.linspace(-padratio*dev4search, padratio*dev4search, ninterval4search):
        currentratioX=padratio+each
        currentpadlengthX=min(min(round(tissuelength*currentratioX),spotMinXtif), wtif/scalefactor-spotCoordsTIF[:,0].max())
        currentdeltaX=abs((round(spotMinXtif*scalefactor)-round((spotMinXtif-currentpadlengthX)*scalefactor))-round(currentpadlengthX*scalefactor))
        if currentdeltaX<bestdeltaX:
            bestdeltaX=currentdeltaX
            bestratioX=currentratioX                  
        currentratioY=padratio+each
        currentpadlengthY=min(min(round(tissuelength*currentratioY),spotMinYtif), htif/scalefactor-spotCoordsTIF[:,1].max())
        currentdeltaY=abs((round(spotMinYtif*scalefactor)-round((spotMinYtif-currentpadlengthY)*scalefactor))-round(currentpadlengthY*scalefactor))
        if currentdeltaY<bestdeltaY:
            bestdeltaY=currentdeltaY
            bestratioY=currentratioY                  
    return [bestratioX,bestratioY]    
    
    
def getSTimg(adata,spatial_key,library_id,img_key="hires"):
    return Image.fromarray((adata.uns[spatial_key][library_id]['images'][img_key]*255).astype(np.uint8))
    
def optPadlength2(tissuelength, padratio, spotCoordsTIF, scale_imgIntensity,scalefactor, dev4search=0.2, ninterval4search=50):
    spotMinXtif=spotCoordsTIF[:,0].min()
    spotMinYtif=spotCoordsTIF[:,1].min()
    htif,wtif,ncolor=scale_imgIntensity.shape

    bestratioX=padratio
    #print((spotMinXtif,spotMinYtif))
    bestpadlengthX=min(min(round(tissuelength*bestratioX),spotMinXtif), wtif/scalefactor-spotCoordsTIF[:,0].max() ) #padding length
    newMinX=spotCoordsTIF[:,0].min()-bestpadlengthX # col
    bestdeltaX=np.sum(np.abs((np.array(spotCoordsTIF[:,0])-newMinX)*scalefactor-np.round((np.array(spotCoordsTIF[:,0])-newMinX)*scalefactor)))

    bestratioY=padratio
    bestpadlengthY=min(min(round(tissuelength*bestratioY),spotMinYtif), htif/scalefactor-spotCoordsTIF[:,1].max()) #padding length
    newMinY=spotCoordsTIF[:,1].min()-bestpadlengthY # col
    bestdeltaY=np.sum(np.abs((np.array(spotCoordsTIF[:,1])-newMinY)*scalefactor-np.round((np.array(spotCoordsTIF[:,1])-newMinY)*scalefactor)))
    for each in np.linspace(-padratio*dev4search, padratio*dev4search, ninterval4search):
        currentratioX=padratio+each
        currentpadlengthX=min(min(round(tissuelength*currentratioX),spotMinXtif), wtif/scalefactor-spotCoordsTIF[:,0].max())
        newMinX=spotMinXtif-currentpadlengthX 
        currentdeltaX=np.sum(np.abs((np.array(spotCoordsTIF[:,0])-newMinX)*scalefactor-np.round((np.array(spotCoordsTIF[:,0])-newMinX)*scalefactor)))
        if currentdeltaX<bestdeltaX:
            bestdeltaX=currentdeltaX
            bestratioX=currentratioX         
            
        currentratioY=padratio+each
        currentpadlengthY=min(min(round(tissuelength*currentratioY),spotMinYtif), htif/scalefactor-spotCoordsTIF[:,1].max())
        newMinY=spotMinYtif-currentpadlengthY
        currentdeltaY=np.sum(np.abs((np.array(spotCoordsTIF[:,1])-newMinY)*scalefactor-np.round((np.array(spotCoordsTIF[:,1])-newMinY)*scalefactor)))
        if currentdeltaY<bestdeltaY:
            bestdeltaY=currentdeltaY
            bestratioY=currentratioY   
            
    return [bestratioX,bestratioY]    

def optPadlength3(tissuelength, padratio, spotCoordsTIF,scale_imgIntensity, scalefactor, dev4search=0.2, ninterval4search=50):
    refIntensity=np.array([scale_imgIntensity[
        min(scale_imgIntensity.shape[0]-1,round(spotCoordsTIF[x,1]*scalefactor)),
        min(scale_imgIntensity.shape[1]-1,round(spotCoordsTIF[x,0]*scalefactor)),:] 
                for x in range(spotCoordsTIF.shape[0])]).reshape(spotCoordsTIF.shape[0],3)
    htif,wtif,ncolor=scale_imgIntensity.shape

    spotMinXtif=spotCoordsTIF[:,0].min()
    spotMinYtif=spotCoordsTIF[:,1].min()
    bestspotCoords=spotCoordsTIF.copy()
    
    bestratioX=padratio
    bestpadlengthX=min(min(round(tissuelength*bestratioX),spotMinXtif), wtif/scalefactor-spotCoordsTIF[:,0].max() ) #padding length
    newMinX=bestspotCoords[:,0].min()-bestpadlengthX 
    newMaxX=bestspotCoords[:,0].max()+bestpadlengthX 
    hnewMinX=round(newMinX*scalefactor)+0 #PIL Image cropping is 0-index and end-exclusive
    hnewMaxX=round(newMaxX*scalefactor)+1
    bestspotCoords[:,0]=bestspotCoords[:,0]-newMinX
    
    bestratioY=padratio
    bestpadlengthY=min(min(round(tissuelength*bestratioY),spotMinYtif) , htif/scalefactor-spotCoordsTIF[:,1].max())#padding length
    newMinY=bestspotCoords[:,1].min()-bestpadlengthY # row
    newMaxY=bestspotCoords[:,1].max()+bestpadlengthY 
    hnewMinY=round(newMinY*scalefactor)+0
    hnewMaxY=round(newMaxY*scalefactor)+1
    bestspotCoords[:,1]=bestspotCoords[:,1]-newMinY
    
    hcropped=Image.fromarray((scale_imgIntensity.copy()* 255).astype(np.uint8)).convert('RGB').crop((hnewMinX,hnewMinY,hnewMaxX,hnewMaxY))
    hcropped=(np.asarray(hcropped)/255).astype(np.float32)
    bestIntensity=np.array([hcropped[
        min(hcropped.shape[0]-1,round(bestspotCoords[x,1]*scalefactor)),
        min(hcropped.shape[1]-1,round(bestspotCoords[x,0]*scalefactor)),:] 
                for x in range(bestspotCoords.shape[0])]).reshape(bestspotCoords.shape[0],3)
    bestdelta=np.sum(np.abs(bestIntensity-refIntensity))
    for eachx in np.linspace(-padratio*dev4search, padratio*dev4search, ninterval4search):
        for eachy in np.linspace(-padratio*dev4search, padratio*dev4search, ninterval4search):
            currentspotCoords=spotCoordsTIF.copy()
            currentratioX=padratio+eachx
            currentpadlengthX=min(min(round(tissuelength*currentratioX),spotMinXtif),wtif/scalefactor-spotCoordsTIF[:,0].max() ) #padding length
            newMinX=currentspotCoords[:,0].min()-currentpadlengthX # col
            newMaxX=currentspotCoords[:,0].max()+currentpadlengthX # col
            hnewMinX=round(newMinX*scalefactor)+0 #PIL Image cropping is 0-index and end-exclusive
            hnewMaxX=round(newMaxX*scalefactor)+1
            currentspotCoords[:,0]=currentspotCoords[:,0]-newMinX

            currentratioY=padratio+eachy
            currentpadlengthY=min(min(round(tissuelength*currentratioY),spotMinYtif),htif/scalefactor-spotCoordsTIF[:,1].max()) #padding length
            newMinY=currentspotCoords[:,1].min()-currentpadlengthY # row
            newMaxY=currentspotCoords[:,1].max()+currentpadlengthY # row
            hnewMinY=round(newMinY*scalefactor)+0
            hnewMaxY=round(newMaxY*scalefactor)+1
            currentspotCoords[:,1]=currentspotCoords[:,1]-newMinY
            
            hcropped=Image.fromarray((scale_imgIntensity.copy()* 255).astype(np.uint8)).convert('RGB').crop((hnewMinX,hnewMinY,hnewMaxX,hnewMaxY))
            hcropped=(np.asarray(hcropped)/255).astype(np.float32)
            currentIntensity=np.array([hcropped[
                min(hcropped.shape[0]-1,round(currentspotCoords[x,1]*scalefactor)),
                min(hcropped.shape[1]-1,round(currentspotCoords[x,0]*scalefactor)),:] 
                                     for x in range(currentspotCoords.shape[0])]).reshape(currentspotCoords.shape[0],3)

            currentdelta=np.sum(np.abs(currentIntensity-refIntensity))

            if currentdelta<bestdelta:
                bestdelta=currentdelta
                bestratioX=currentratioX                  
                bestratioY=currentratioY                  
    return [bestratioX,bestratioY]    

def optPadlength4(tissuelength, padratio, spotCoordsTIF,scale_imgIntensity, scalefactor, dev4search=0.2, ninterval4search=50):
    org_hires=Image.fromarray((scale_imgIntensity* 255).astype(np.uint8)).convert('RGB')
    refIntensity=np.array([scale_imgIntensity[round(spotCoordsTIF[x,1]*scalefactor),
                                     round(spotCoordsTIF[x,0]*scalefactor),:] 
                for x in range(spotCoordsTIF.shape[0])]).reshape(spotCoordsTIF.shape[0],3)
    htif,wtif,ncolor=scale_imgIntensity.shape

    spotMinXtif=spotCoordsTIF[:,0].min()
    spotMinYtif=spotCoordsTIF[:,1].min()
    bestspotCoords=spotCoordsTIF.copy()

    bestratioX=padratio
    bestpadlengthX=round(tissuelength*bestratioX) #padding length
    newMinX=bestspotCoords[:,0].min()-bestpadlengthX #col/width/left
    #newMaxX=bestspotCoords[:,0].max()+bestpadlengthX
    himgMinX=round(abs(newMinX)*scalefactor)+0
    hnew_width = round((bestspotCoords[:,0].max()-bestspotCoords[:,0].min()+1+2*bestpadlengthX)*scalefactor)
    bestspotCoords[:,0]=bestspotCoords[:,0]-newMinX

    bestratioY=padratio
    bestpadlengthY=round(tissuelength*bestratioY) 
    newMinY=bestspotCoords[:,1].min()-bestpadlengthY # row
    #newMaxY=bestspotCoords[:,1].max()+bestpadlengthY 
    himgMinY=round(abs(newMinY)*scalefactor)+0
    hnew_height = round((bestspotCoords[:,1].max()-bestspotCoords[:,1].min()+1+2*bestpadlengthY)*scalefactor)
    bestspotCoords[:,1]=bestspotCoords[:,1]-newMinY
    
    hpadded=Image.new("RGB", (hnew_width, hnew_height), (0, 0, 0))
    hpadded.paste(org_hires,(himgMinX,himgMinY))
    hpadded=(np.asarray(hpadded)/255).astype(np.float32)
    #print(hpadded.shape)
    #print([round(bestspotCoords[:,1].max()*scalefactor),round(bestspotCoords[:,0].max()*scalefactor)])

    bestIntensity=np.array([hpadded[round(bestspotCoords[x,1]*scalefactor),
                                     round(bestspotCoords[x,0]*scalefactor),:] 
                for x in range(bestspotCoords.shape[0])]).reshape(bestspotCoords.shape[0],3)
    bestdelta=np.sum(np.abs(bestIntensity-refIntensity))
    for eachx in np.linspace(-padratio*dev4search, padratio*dev4search, ninterval4search):
        for eachy in np.linspace(-padratio*dev4search, padratio*dev4search, ninterval4search):
            currentspotCoords=spotCoordsTIF.copy()
            
            currentratioX=padratio+eachx
            currentpadlengthX=round(tissuelength*currentratioX)  #padding length
            newMinX=currentspotCoords[:,0].min()-currentpadlengthX # col
            #newMaxX=currentspotCoords[:,0].max()+currentpadlengthX # col
            himgMinX=round(abs(newMinX)*scalefactor)+0
            hnew_width = round((currentspotCoords[:,0].max()-currentspotCoords[:,0].min()+1+2*currentpadlengthX)*scalefactor)
            currentspotCoords[:,0]=currentspotCoords[:,0]-newMinX

            currentratioY=padratio+eachy
            currentpadlengthY=round(tissuelength*currentratioY) 
            newMinY=currentspotCoords[:,1].min()-currentpadlengthY # row
            #newMaxY=currentspotCoords[:,1].max()+currentpadlengthY # row
            himgMinY=round(abs(newMinY)*scalefactor)+0
            hnew_height = round((currentspotCoords[:,1].max()-currentspotCoords[:,1].min()+1+2*currentpadlengthY)*scalefactor)
            currentspotCoords[:,1]=currentspotCoords[:,1]-newMinY
            
            hpadded=Image.new("RGB", (hnew_width, hnew_height), (0, 0, 0))
            hpadded.paste(org_hires,(himgMinX,himgMinY))
            hpadded=(np.asarray(hpadded)/255).astype(np.float32)
            currentIntensity=np.array([hpadded[round(currentspotCoords[x,1]*scalefactor),
                                     round(currentspotCoords[x,0]*scalefactor),:] 
                                     for x in range(currentspotCoords.shape[0])]).reshape(currentspotCoords.shape[0],3)

            currentdelta=np.sum(np.abs(currentIntensity-refIntensity))

            if currentdelta<bestdelta:
                bestdelta=currentdelta
                bestratioX=currentratioX                  
                bestratioY=currentratioY     
    return [bestratioX,bestratioY]    


def padAdata(adata,spatial_key,library_id, padratio=0.1, doMask=False, withOuts=False,dev4search=0.2,ninterval4search=50):
    adata_pad = adata.copy()
    newobsm=adata_pad.obsm.copy() # spot coords at full TIFF resolution
    newobs=adata_pad.obs.copy()
    newobsm[spatial_key]=newobsm[spatial_key][list(np.where(newobs['in_tissue'].to_numpy()==1)[0])] # new obsm['spatial']
    newobs=newobs.loc[newobs['in_tissue']==1] #new obs
    scale_imgIntensity=adata_pad.uns[spatial_key][library_id]['images']['hires']
    htif,wtif,ncolor=scale_imgIntensity.shape

    scalelow=scalehigh=1
    if 'tissue_lowres_scalef' in adata_pad.uns[spatial_key][library_id]['scalefactors'].keys():
        scalelow=adata_pad.uns[spatial_key][library_id]['scalefactors']['tissue_lowres_scalef']
    if 'tissue_hires_scalef' in adata_pad.uns[spatial_key][library_id]['scalefactors'].keys():
        scalehigh=adata_pad.uns[spatial_key][library_id]['scalefactors']['tissue_hires_scalef']

    xy_indices=newobsm[spatial_key].copy()
    intissueMinx=(xy_indices[:,0].min())
    intissueMaxx=(xy_indices[:,0].max())
    intissueMiny=(xy_indices[:,1].min())
    intissueMaxy=(xy_indices[:,1].max())

    #@@ change to use tissue boundary not just in_tissue spot boundary to decide padding
    deltax=xy_indices[:,0].max()-xy_indices[:,0].min()
    deltay=xy_indices[:,1].max()-xy_indices[:,1].min()
    tissuelength=max(deltax,deltay)
    #padlength=round(tissuelength*padratio) #padding length

    bestratioX, bestratioY=optPadlength4(tissuelength=tissuelength, padratio=padratio,spotCoordsTIF=xy_indices, 
                                         scale_imgIntensity=scale_imgIntensity,
                                        scalefactor=scalehigh, dev4search=dev4search, ninterval4search=ninterval4search)
    #print((bestratioX, bestratioY))
    bestpadlengthX=round(tissuelength*bestratioX) 
    bestpadlengthY=round(tissuelength*bestratioY) 


    newMinX=xy_indices[:,0].min()-bestpadlengthX # col/width/left
    newMaxX=xy_indices[:,0].max()+bestpadlengthX
    newMinY=xy_indices[:,1].min()-bestpadlengthY # row/height/top
    newMaxY=xy_indices[:,1].max()+bestpadlengthY


    
    #shift spot coords to match newly cropped img
    newobsm[spatial_key][:,0]=newobsm[spatial_key][:,0]-newMinX
    newobsm[spatial_key][:,1]=newobsm[spatial_key][:,1]-newMinY

    # new TIFF start X = newMinX
    # new TIFF start Y = newMinY    
    #convert to new starts in actual hires or lowres images
    himgMinX=round(abs(newMinX)*scalehigh)+0
    himgMinY=round(abs(newMinY)*scalehigh)+0
    
    limgMinX=round(abs(newMinX)*scalelow)+0
    limgMinY=round(abs(newMinY)*scalelow)+0
  
    hnew_width = round((xy_indices[:,0].max()-xy_indices[:,0].min()+1+2*bestpadlengthX)*scalehigh)
    hnew_height = round((xy_indices[:,1].max()-xy_indices[:,1].min()+1+2*bestpadlengthY)*scalehigh)
    lnew_width = round((xy_indices[:,0].max()-xy_indices[:,0].min()+1+2*bestpadlengthX)*scalelow)
    lnew_height = round((xy_indices[:,1].max()-xy_indices[:,1].min()+1+2*bestpadlengthY)*scalelow)
    #hnew_width = hnewMaxX-hnewMinX + 1
    #hnew_height = hnewMaxY-hnewMinY + 1
    #lnew_width = lnewMaxX-lnewMinX + 1
    #lnew_height = lnewMaxY-lnewMinY + 1

    #left, top, right, bottom
    if doMask:
        if 'tissue_lowres_scalef' in adata_pad.uns[spatial_key][library_id]['scalefactors'].keys():
            masked_lowres=maskNonTissue(adata_pad.uns[spatial_key][library_id]['images']['lowres'],intissueMinx,intissueMiny,intissueMaxx,intissueMaxy,None)
            masked_lowres=Image.fromarray((masked_lowres*255).astype(np.uint8)).convert('RGB')
            lpadded=Image.new("RGB", (lnew_width, lnew_height), (0, 0, 0))
            lpadded.paste(masked_lowres,(limgMinX,limgMinY))
        masked_hires=maskNonTissue(adata_pad.uns[spatial_key][library_id]['images']['hires'],intissueMinx,intissueMiny,intissueMaxx,intissueMaxy,None)
        masked_hires=Image.fromarray((masked_hires*255).astype(np.uint8)).convert('RGB')
        hpadded=Image.new("RGB", (hnew_width, hnew_height), (0, 0, 0))
        hpadded.paste(masked_hires,(himgMinX,himgMinY))
    else:
        if 'tissue_lowres_scalef' in adata_pad.uns[spatial_key][library_id]['scalefactors'].keys():
            org_lowres=Image.fromarray((adata_pad.uns[spatial_key][library_id]['images']['lowres'] * 255).astype(np.uint8)).convert('RGB')
            lpadded=Image.new("RGB", (lnew_width, lnew_height), (0, 0, 0))
            Image.Image.paste(lpadded,org_lowres,(limgMinX,limgMinY))
        org_hires=Image.fromarray((adata_pad.uns[spatial_key][library_id]['images']['hires'] * 255).astype(np.uint8)).convert('RGB')
        hpadded=Image.new("RGB", (hnew_width, hnew_height), (0, 0, 0))
        hpadded.paste(org_hires,(himgMinX,himgMinY))

    if 'tissue_lowres_scalef' in adata_pad.uns[spatial_key][library_id]['scalefactors'].keys():
        adata_pad.uns[spatial_key][library_id]['images']['lowres']=(np.asarray(lpadded)/255).astype(np.float32)
    adata_pad.uns[spatial_key][library_id]['images']['hires']=(np.asarray(hpadded)/255).astype(np.float32)
    adata_pad.obs=newobs
    adata_pad.obsm=newobsm
    if withOuts:
        return [adata_pad, {'padlength':[bestpadlengthX,bestpadlengthY],'scalehigh':scalehigh,'tissuelength':tissuelength,'hnew_width':hnew_width,'hnew_height':hnew_height,
                        'himgMinX':himgMinX, 'himgMinY':himgMinY,
                       'newMinX':newMinX,'newMaxX':newMaxX,'newMinY':newMinY,'newMaxY':newMaxY}]
    else:
        return adata_pad

def maskNonTissue(imgarray,minx,miny,maxx,maxy,scaling=None,connectivity=4,rounds=1):
    test=imgarray.copy() #full image before cropping
    mask=standard_mask_fn(Image.fromarray((test* 255).astype(np.uint8)).convert('RGB'),max_mask_dim=2000,connectivity=connectivity,rounds=rounds)
    test[~np.array(mask)]=0
    if scaling is not None:
        minx=round(minx*scaling)
        miny=round(miny*scaling)
        maxx=round(maxx*scaling)
        maxy=round(maxy*scaling)
        test[0:minx,:,:]=0
        test[(maxx+1):-1,:,:]=0
        test[:,0:miny,:]=0
        test[:,(maxy+1):-1,:]=0
    return test

    
def encodeLBL(lbl_map,classes=None):
    elbl_map=np.zeros((lbl_map.shape[0],lbl_map.shape[1],np.unique(lbl_map).size))
    if classes is not None:
        classes=np.sort(np.unique(lbl_map)).astype(np.float64).tolist()
    for each in range(len(classes)):
        elbl_map[lbl_map==classes[each],each]=1
    return elbl_map

## follow the voronoi rule to assign label to each pixel
def dilateLBL(spot_csv,stobj,resolution="maincluster",offset=100,nonTissueLabel=0, missingLabel=-1,res="hires"):
    #offset: systematically shift cluster label a value range that is more image friendly
    spot_coords=pd.DataFrame(stobj.obsm['spatial'].copy(),index=stobj.obs.index)
    spot_coords=spot_coords.loc[list(spot_csv['spotID'])]
    spatial_id=list(stobj.uns.keys())[0]
    library_id=list(stobj.uns['spatial'].keys())[0]
    maxH=stobj.uns[spatial_id][library_id]['images'][res].shape[0]
    maxW=stobj.uns[spatial_id][library_id]['images'][res].shape[1]
    scale_factor=stobj.uns[spatial_id][library_id]['scalefactors']['tissue_'+res+'_scalef']

    spot_coords=spot_coords*scale_factor
    spot_coords[0]=[int(round(x,0)) for x in spot_coords[0].tolist()]
    spot_coords[1]=[int(round(x,0)) for x in spot_coords[1].tolist()]
    spot_coords.drop_duplicates(inplace=True)
    spot_csv=spot_csv.loc[spot_coords.index.tolist()]
    scaled_spot_coords=spot_coords.to_numpy()
    scaled_spot_coords=scaled_spot_coords.astype(np.float32)
    
    # xs = range(maxW)
    # ys = range(maxH)
    # queryspot = np.array(list(product(xs, ys)))
    
    #@ pass in the tissue mask to be safer 
    xkeep,ykeep = np.where(np.transpose(stobj.uns[spatial_id][library_id]['images'][res][:,:,0])!=0)
    queryspot=pd.DataFrame({'0':xkeep,'1':ykeep}).to_numpy()
    lbl_map=np.zeros((maxW,maxH))
    transfered=[]
    
    refspot=scaled_spot_coords
    reflbl=spot_csv
    #node_dict = dict(zip(range(refspot.shape[0]), reflbl.index))
    node_dict=pd.DataFrame(reflbl.index.tolist(),index=range(refspot.shape[0]))

    knn=0 
    batchsize=10000
    counter=0
    while counter<queryspot.shape[0]:
        miniqueryspot=queryspot[counter:min((counter+batchsize),queryspot.shape[0]),:]        
        D=distance_matrix(miniqueryspot,refspot)
        #@ todo: enable randomization to break ties
        idx = np.argsort(D, 1)[:, 0:knn+1]
       # b=np.random.random(D.shape[1])
        #idx = np.lexsort((D,), 1)[:, 0:knn+1]    
        tmp=(reflbl.loc[node_dict.loc[idx[:,0].tolist()][0]][resolution]+offset).tolist()
        transfered.extend(tmp)
        #print(counter)
        counter=counter+batchsize
    
    lbl_map[queryspot[:,0],queryspot[:,1]]=transfered
    lbl_map=lbl_map.reshape((maxW,maxH))
    # lbl_map=np.zeros((maxW,maxH))
    # for y in ykeep.tolist():
    #     for x in xkeep.tolist():
    # for y in range(maxH):
    #     for x in range(maxW):
    #         new_point=np.array([[x, y]])
    #         # find nearest spot and break tie randomly
    #         dist=np.sum((scaled_spot_coords - new_point)**2, axis=1)
    #         whichSpot=np.random.choice(np.where(dist == dist.min())[0])
    #         lbl_map[x,y]=spot_csv.iloc[whichSpot][resolution]+offset
    
    #@ pass in the tissue mask to be safer 
    lbl_map[np.transpose(stobj.uns[spatial_id][library_id]['images'][res][:,:,0])==0]=nonTissueLabel
    return lbl_map


def arr2Img(imgarray):
    if imgarray.max()<=1:
        imgarray=(imgarray*255)
    return Image.fromarray(imgarray.astype(np.uint8)).convert("RGB")



def getCountourImage2(seg_image):
    filters = []
    for i in [0, 1, 2]:
     for j in [0, 1, 2]:
        filter = np.zeros([3,3], dtype=np.int)
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

from PIL import ImageFilter
import scipy
import skimage

def makeContourImg(adata_mini_pad_warped_d5,lbl_map_mini_warped,tomerge,maskval=0,med_filter_size=4,trimMax=3,res="hires", smooth=False):
    spatial_key=list(adata_mini_pad_warped_d5.uns.keys())[0]
    library_id=list(adata_mini_pad_warped_d5.uns[spatial_key].keys())[0]

    scale_factor=adata_mini_pad_warped_d5.uns[spatial_key][library_id]['scalefactors']['tissue_'+res+'_scalef']
    rowMin=adata_mini_pad_warped_d5.obsm[spatial_key][:,1].min()*scale_factor
    rowMax=adata_mini_pad_warped_d5.obsm[spatial_key][:,1].max()*scale_factor
    colMin=adata_mini_pad_warped_d5.obsm[spatial_key][:,0].min()*scale_factor
    colMax=adata_mini_pad_warped_d5.obsm[spatial_key][:,0].max()*scale_factor
    rowDelta=(rowMax-rowMin)*1.0/math.sqrt(len(adata_mini_pad_warped_d5.obsm[spatial_key][:,1]))
    colDelta=(colMax-colMin)*1.0/math.sqrt(len(adata_mini_pad_warped_d5.obsm[spatial_key][:,0]))
    
    test=lbl_map_mini_warped.copy()
    test[0:math.floor(max(1,rowMin-rowDelta)),:]=0
    test[math.ceil(min(lbl_map_mini_warped.shape[0]-1,rowMax+rowDelta)):-1,:]=0
    test[:,0:math.floor(max(1,colMin-colDelta))]=0
    test[:,math.ceil(min(lbl_map_mini_warped.shape[1]-1,colMax+colDelta)):-1]=0
    cp_warped=test.copy()
    for i in range(len(tomerge)):
        ref=tomerge[i][0]
        for j in tomerge[i]:
            test[np.where(test==j)]=ref
    #test[np.where((test==102)|(test==106)|(test==104)|(test==121)|(test==111))]=102
    #test[np.where((test==117)|(test==118))]=117
    #Image.fromarray((test).astype(np.uint8)).show()
    lbl_map_mini_warped=test

    contour2=getCountourImage2(lbl_map_mini_warped)
    contourimg2=np.zeros((lbl_map_mini_warped.shape[0],lbl_map_mini_warped.shape[1],3))
    contourimg2[:,:,0]=contour2
    contourimg2[:,:,1]=contour2
    contourimg2[:,:,2]=contour2
    #print(np.unique(contour2))
    contourimg2=scipy.ndimage.median_filter((contourimg2).astype(np.uint8), size=med_filter_size)
    
    contourimg2=contourimg2/255
    
    tissueEdge_warped=np.where((cp_warped==maskval)&(contourimg2[:,:,0]>0))

    arm0=np.array(tissueEdge_warped[0])
    arm1=np.array(tissueEdge_warped[1])
    for i in range(tissueEdge_warped[0].shape[0]):
     for s in range(-trimMax,trimMax+1):
        for t in range(-trimMax,trimMax+1):
            if tissueEdge_warped[0][i]+s>=0 and tissueEdge_warped[0][i]+s<lbl_map_mini_warped.shape[0] and \
                tissueEdge_warped[1][i]+t>=0 and tissueEdge_warped[1][i]+t<lbl_map_mini_warped.shape[1]:
                 arm0=np.append(arm0,tissueEdge_warped[0][i]+s)
                 arm1=np.append(arm1,tissueEdge_warped[1][i]+t)

    tissueEdge_warped_add=(arm0,arm1)
    #print(arm0.shape)
    contourimg2[tissueEdge_warped_add]=0
    
    if smooth:
     footprint = skimage.morphology.disk(3)
     res = skimage.morphology.closing(contourimg2[:,:,0],footprint)

     footprint = skimage.morphology.disk(1)
     res = skimage.morphology.dilation(res,footprint)

     footprint = skimage.morphology.disk(2)
     res = skimage.morphology.opening(res,footprint)
     res = skimage.morphology.opening(res,footprint)
     res = skimage.morphology.opening(res,footprint)
     res = skimage.morphology.opening(res,footprint)
     res = skimage.morphology.opening(res,footprint)
    

     footprint = skimage.morphology.disk(1)
     res = skimage.morphology.erosion(res,footprint)
     footprint = skimage.morphology.disk(1)
     res = skimage.morphology.erosion(res,footprint)
     footprint = skimage.morphology.disk(1)
     res = skimage.morphology.dilation(res,footprint)
    
    # apply Gaussian blur
#     res = skimage.filters.gaussian(res,
#         sigma=(1, 1), truncate=1, channel_axis=-1)
    
     contourimg2[:,:,0]=res
     contourimg2[:,:,1]=res
     contourimg2[:,:,2]=res
    
    return lbl_map_mini_warped,contourimg2



    
def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    smooth = 0.0001
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def dice_coef_multilabel(y_true, y_pred, numLabels):
    dice=0
    for index in range(numLabels):
        dice += dice_coef(y_true[:,:,:,index], y_pred[:,:,:,index])
    return dice/numLabels # taking average

def iou_coef(y_true, y_pred, smooth=1):
  intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
  union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
  iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
  return iou


def cropAdataByHires(adata,bestpadlengthX,bestpadlengthY, finalhires=None):
    adata_cp=adata.copy()
    spatial_key=list(adata_cp.uns.keys())[0]
    library_key=list(adata_cp.uns[spatial_key].keys())[0]
    scalehigh=adata_cp.uns[spatial_key][library_key]['scalefactors']['tissue_'+str('hires')+'_scalef']        
    himgX=adata_cp.uns[spatial_key][library_key]['images']['hires'].shape[0]
    himgY=adata_cp.uns[spatial_key][library_key]['images']['hires'].shape[1]
    limgX=adata_cp.uns[spatial_key][library_key]['images']['hires'].shape[0]
    limgY=adata_cp.uns[spatial_key][library_key]['images']['hires'].shape[1]
    
    xy_indices=adata_cp.obsm['spatial'].copy()
    x0=xy_indices[:,0].min()
    y0=xy_indices[:,1].min()
    xM=xy_indices[:,0].max()
    yM=xy_indices[:,1].max()        

    newMinX=x0-bestpadlengthX # col?
    newMinY=y0-bestpadlengthY # row?
    newMaxX=xM+bestpadlengthX
    newMaxY=yM+bestpadlengthY
    
    if finalhires is not None:
        #if round(newMaxX*scalehigh)+1>round(newMinX*scalehigh)+finalhires:
        if round(xM*scalehigh)+1>round(x0*scalehigh)+finalhires*1.02 or round(yM*scalehigh)+1>round(y0*scalehigh)+finalhires*1.02:
            print("ERROR: image width is too large to fit finalhires without cropping out tissue")
            return None

    #shift spot coords to match newly cropped img
    xy_indices[:,0]=xy_indices[:,0]-newMinX # = padlength
    xy_indices[:,1]=xy_indices[:,1]-newMinY
    hnewMinX=round(newMinX*scalehigh)+0 #PIL Image cropping is 0-index and end-exclusive
    hnewMinY=round(newMinY*scalehigh)+0
    hnewMaxX=round(newMaxX*scalehigh)+1
    hnewMaxY=round(newMaxY*scalehigh)+1
    if finalhires is not None:
        #round(newMaxX*scalehigh)+1-round(newMinX*scalehigh)=finalhires
        hnewMaxX=round(newMinX*scalehigh)+finalhires
        hnewMaxY=round(newMinY*scalehigh)+finalhires
    hcropped=Image.fromarray((adata_cp.uns[spatial_key][library_key]['images']['hires']* 255).astype(np.uint8)).convert('RGB').crop((hnewMinX,hnewMinY,hnewMaxX,hnewMaxY))
    adata_cp.uns[spatial_key][library_key]['images']['hires']=(np.asarray(hcropped)/255).astype(np.float32)

    if 'tissue_'+str('lowres')+'_scalef' in list(adata_cp.uns[spatial_key][library_key]['scalefactors'].keys()):
        scalelow=adata_cp.uns[spatial_key][library_key]['scalefactors']['tissue_'+str('lowres')+'_scalef']
        lnewMinX=round(newMinX*scalelow)+0 #PIL Image cropping is 0-index and end-exclusive
        lnewMaxX=round(newMaxX*scalelow)+1
        lnewMinY=round(newMinY*scalelow)+0
        lnewMaxY=round(newMaxY*scalelow)+1
        if finalhires is not None:
            lnewMaxX=round(newMinX*scalelow)+round(finalhires*scalelow/scalehigh)
            lnewMaxY=round(newMinY*scalelow)+round(finalhires*scalelow/scalehigh)
        lcropped=Image.fromarray((adata_cp.uns[spatial_key][library_key]['images']['lowres']* 255).astype(np.uint8)).convert('RGB').crop((lnewMinX,lnewMinY,lnewMaxX,lnewMaxY))
        adata_cp.uns[spatial_key][library_key]['images']['lowres']=(np.asarray(lcropped)/255).astype(np.float32)

    adata_cp.obs=adata_cp.obs.loc[adata_cp.obs['in_tissue']==1] #new obs
    adata_cp.obsm['spatial']=xy_indices
    return adata_cp

def padAdataByHires(adata,bestpadlengthX,bestpadlengthY, finalhires=None):
    adata_cp=adata.copy()
    spatial_key=list(adata_cp.uns.keys())[0]
    library_key=list(adata_cp.uns[spatial_key].keys())[0]
    scalehigh=adata_cp.uns[spatial_key][library_key]['scalefactors']['tissue_'+str('hires')+'_scalef']   

    xy_indices=adata_cp.obsm['spatial'].copy()
    padX4FinalHires=0
    padY4FinalHires=0
    if finalhires is not None:
            padX4FinalHires=(finalhires/scalehigh-(xy_indices[:,0].max()-xy_indices[:,0].min()+1))/2
            padY4FinalHires=(finalhires/scalehigh-(xy_indices[:,1].max()-xy_indices[:,1].min()+1))/2
    bestpadlengthX=max(padX4FinalHires, bestpadlengthX)#padding length
    bestpadlengthY=max(padY4FinalHires, bestpadlengthY) #padding length
        
    x0=xy_indices[:,0].min()
    y0=xy_indices[:,1].min()
    xM=xy_indices[:,0].max()
    yM=xy_indices[:,1].max()        

    newMinX=x0-bestpadlengthX # col?
    newMinY=y0-bestpadlengthY # row?
    
    #shift spot coords to match newly cropped img
    xy_indices[:,0]=xy_indices[:,0]-newMinX
    xy_indices[:,1]=xy_indices[:,1]-newMinY
    
#    himgMinX=round(abs(newMinX)*scalehigh)+0
#    himgMinY=round(abs(newMinY)*scalehigh)+0
    himgMinX=round(-(newMinX)*scalehigh)+0
    himgMinY=round(-(newMinY)*scalehigh)+0
    hnew_width = round((xy_indices[:,0].max()-xy_indices[:,0].min()+1+2*bestpadlengthX)*scalehigh)
    hnew_height = round((xy_indices[:,1].max()-xy_indices[:,1].min()+1+2*bestpadlengthY)*scalehigh)
    if finalhires is not None:
        hnew_width=max(hnew_width,finalhires)
        hnew_height=max(hnew_height,finalhires)

    hpadded=Image.new("RGB", (hnew_width, hnew_height), (0, 0, 0))
    orgimg=Image.fromarray((adata_cp.uns[spatial_key][library_key]['images']['hires'] * 255).astype(np.uint8)).convert('RGB')
    hpadded.paste(orgimg,(himgMinX,himgMinY))
    adata_cp.uns[spatial_key][library_key]['images']['hires']=(np.asarray(hpadded)/255).astype(np.float32)

    if 'tissue_'+str('lowres')+'_scalef' in list(adata_cp.uns[spatial_key][library_key]['scalefactors'].keys()):
        scalelow=adata_cp.uns[spatial_key][library_key]['scalefactors']['tissue_'+str('lowres')+'_scalef']
#        limgMinX=round(abs(newMinX)*scalelow)+0
#        limgMinY=round(abs(newMinY)*scalelow)+0  
        limgMinX=round(-(newMinX)*scalelow)+0
        limgMinY=round(-(newMinY)*scalelow)+0  
        lnew_width = round((xy_indices[:,0].max()-xy_indices[:,0].min()+1+2*bestpadlengthX)*scalelow)
        lnew_height = round((xy_indices[:,1].max()-xy_indices[:,1].min()+1+2*bestpadlengthY)*scalelow)
        if finalhires is not None:
            lnew_width=max(lnew_width,round(finalhires*scalelow/scalehigh))
            lnew_height=max(lnew_height,round(finalhires*scalelow/scalehigh))
        lpadded=Image.new("RGB", (lnew_width, lnew_height), (0, 0, 0))
        orgimg=Image.fromarray((adata_cp.uns[spatial_key][library_key]['images']['lowres'] * 255).astype(np.uint8)).convert('RGB')
        lpadded.paste(orgimg,(limgMinX,limgMinY))
        adata_cp.uns[spatial_key][library_key]['images']['lowres']=(np.asarray(lpadded)/255).astype(np.float32)

    adata_cp.obs=adata_cp.obs.loc[adata_cp.obs['in_tissue']==1] #new obs
    adata_cp.obsm['spatial']=xy_indices
    return adata_cp


def prep4AF(im_mov, mask_mov=None,max_reg_dim=512,defaultvalue=0, mask_fn=None):
    #@@ simplify to extract mask_mov/fix from the input img directly
    if mask_fn is None:
        #mask_fn = lambda pil_im: standard_mask_fn(pil_im=pil_im,max_mask_dim=max_reg_dim) 
        mask_fn=lambda pil_img: Image.fromarray(np.ones(pil_img.size[::-1], dtype=bool))
    # Preprocess and verify the images.
    if not isinstance(im_mov, Image.Image): im_mov = Image.fromarray(im_mov)
    #if not isinstance(im_mov, Image.Image): im_fix = Image.fromarray(im_fix)
    #assert np.array_equal(im_mov.size, im_fix.size), 'Anndata objects must reference images of the same size.'
    # Preprocess and verify the masks if a masking function present.
    if mask_mov is not None:
        if not isinstance(mask_mov, Image.Image): mask_mov = Image.fromarray(mask_mov)
    elif mask_fn is not None: mask_mov = mask_fn(im_mov)
    assert np.array_equal(im_mov.size, mask_mov.size) and mask_mov.mode == '1', \
            'mask_mov must be a binary PIL mask with the same size as im_mov.'
    #if mask_fix is not None:
    #    if not isinstance(mask_fix, Image.Image): mask_fix = Image.fromarray(mask_fix)
    #elif mask_fn is not None: mask_fix = mask_fn(im_fix)
    #assert np.array_equal(im_fix.size, mask_fix.size) and mask_fix.mode == '1', \
    #        'mask_fix must be a binary PIL mask with the same size as im_fix.'
    # APPLY MASKS TO INITIAL IMAGES
    im_mov = np.array(im_mov)
    im_mov[~np.asarray(mask_mov).astype(bool)] = defaultvalue
    im_mov = Image.fromarray(im_mov)
    return im_mov, mask_mov

def afTransform(im_mov, im_fix, mask_mov, mask_fix,spatial_mov,max_reg_dim=512,defaultvalue=0, search_factor=5 ,use_principal_axis=True, type_of_transform='Affine'):
    
    # make a "small" size for faster af
    im_mov_r, im_fix_r, mask_mov_r, mask_fix_r = map(lambda x, resample: resize_to_max_dim_pil(x,
                                                                                                max_dim=max_reg_dim,
                                                                                                resample=resample),
                                                        (im_mov, im_fix, mask_mov, mask_fix),
                                                        (Image.Resampling.BILINEAR,#
                                                        Image.Resampling.BILINEAR, #
                                                        Image.Resampling.NEAREST,#
                                                        Image.Resampling.NEAREST))#
        
    # perform affine registration; omit itk for now.
    if search_factor>=20:
         #print(' initializer is on')
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

    # upscale the affine transform
    large_aff_tfm = ants.read_transform(aff_tfm_info['fwdtransforms'][0]) # assume only one matrix out.
    large_aff_tfm = scale_affine(trf=large_aff_tfm,
                                        orig_shape_2d=im_mov_r.size[::-1],
                                        new_shape_2d=im_mov.size[::-1])
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

    inv_large_aff_tfm_itk=large_aff_tfm_itk.GetInverse()
    # need to apply inverse transform to mimic image's resample.
    spatial_mov_a = register_spatial_with_itk_points(
                    spatial_data=spatial_mov,
                    inverse_itk_trf=inv_large_aff_tfm_itk, 
                    spatial_data_indexing='xy')
    return im_mov_a, mask_mov_a, spatial_mov_a

   
def scaleAdata(adataset,  ratio=0.07, finalhires=512, coarse=0, onlyCropPad=False, search_factor=20, use_principal_axis=False,type_of_transform='Affine',mask_fn=None):
    res='hires'
    rescaled=[]
    spatial_key=list(adataset[0].uns.keys())[0]
    library_key=list(adataset[0].uns[spatial_key].keys())[0]
    
    if not onlyCropPad:
     af=[]
     for each in range(len(adataset)):
        tmp=adataset[each].copy()
        spatial_mov=tmp.uns[spatial_key].values()
        movd_spatial_nm = next(iter(tmp.uns[spatial_key].keys()))
        im_mov=Image.fromarray((tmp.uns[spatial_key][movd_spatial_nm]['images'][res] * 255).astype(np.uint8)).convert('RGB')
        scalefactor_mov = tmp.uns[spatial_key][movd_spatial_nm]['scalefactors']['tissue_%s_scalef' % res]
        spatial_mov_scaled = tmp.obsm['spatial'] * scalefactor_mov
        [im_mov, mask_mov]=prep4AF(im_mov, mask_mov=None, mask_fn=mask_fn)
        tmp.uns['spatial'][movd_spatial_nm]['images'][res]=im_mov
        tmp.uns['spatial'][movd_spatial_nm]['images']['%s_mask' % res]=mask_mov
        tmp.obsm['spatial'] =spatial_mov_scaled
        af.append(tmp)
        
     rest=set(range(len(adataset))).difference(set([coarse]))
     im_fix=af[coarse].uns[spatial_key][next(iter(af[coarse].uns[spatial_key].keys()))]['images'][res]
     mask_fix=af[coarse].uns[spatial_key][next(iter(af[coarse].uns[spatial_key].keys()))]['images']['%s_mask' % res]
     for each in rest:
        movd_spatial_nm=next(iter(af[each].uns[spatial_key].keys()))
        im_mov=af[each].uns[spatial_key][movd_spatial_nm]['images'][res]
        mask_mov=af[each].uns[spatial_key][movd_spatial_nm]['images']['%s_mask' % res]
        spatial_mov=af[each].obsm['spatial'] 
        [im_mov_a,mask_mov_a, spatial_mov_a]=afTransform(im_mov, im_fix, mask_mov, mask_fix,spatial_mov,max_reg_dim=256, search_factor=search_factor,
                                                       use_principal_axis=use_principal_axis,type_of_transform=type_of_transform)
        af[each].uns['spatial'][movd_spatial_nm]['images'][res]=im_mov_a
        af[each].obsm['spatial']=spatial_mov_a
        
     for each in range(len(af)):
        movd_spatial_nm=next(iter(af[each].uns[spatial_key].keys()))
        af[each].uns[spatial_key][movd_spatial_nm]['scalefactors']['tissue_%s_scalef' % 'hires']=1
        af[each].uns['spatial'][movd_spatial_nm]['images'][res] = np.asarray(af[each].uns['spatial'][movd_spatial_nm]['images'][res]) / 255
        af[each].obsm['spatial']= af[each].obsm['spatial'] #/ scalefactor_mov
        del af[each].uns[spatial_key][movd_spatial_nm]['images']['%s_mask' % res] #remove  mask to be sure 
    
    #first scaling
     rangeX=np.median([each.obsm['spatial'][:,0].max()-each.obsm['spatial'][:,0].min()+1 for each in af] )
     rangeY=np.median([each.obsm['spatial'][:,1].max()-each.obsm['spatial'][:,1].min()+1 for each in af] )
     scalings=[min(rangeX/(each.obsm['spatial'][:,0].max()-each.obsm['spatial'][:,0].min()+1),rangeY/(each.obsm['spatial'][:,1].max()-each.obsm['spatial'][:,1].min()+1)) for each in af]
    
    
     for each in range(len(af)):
        tmp=af[each].copy()
        movd_spatial_nm = next(iter(tmp.uns[spatial_key].keys()))
        img=Image.fromarray((af[each].uns[spatial_key][movd_spatial_nm]['images'][res]*255).astype(np.uint8))
        img=img.resize([round(dim*scalings[each]) for dim in img.size])
        imgarray=np.array(img)/255.0
        tmp.uns[spatial_key][library_key]['images'][res]=imgarray
        tmp.obsm['spatial']=scalings[each]*tmp.obsm['spatial']
        rescaled.append(tmp)
        
     #print("rangeX="+str(rangeX))
     #print([each.obsm['spatial'][:,0].max()-each.obsm['spatial'][:,0].min()+1 for each in af])
     #print([each.obsm['spatial'][:,0].max()-each.obsm['spatial'][:,0].min()+1 for each in rescaled])
     #print("rangeY="+str(rangeY))
     #print([each.obsm['spatial'][:,1].max()-each.obsm['spatial'][:,1].min()+1 for each in af])
     #print([each.obsm['spatial'][:,1].max()-each.obsm['spatial'][:,1].min()+1 for each in rescaled])
    else: # if only need to crop and pad to reach final resolution
     rescaled=[x.copy() for x in adataset]   

    # update ranges and tissuelength with rescaled 
    rangeX=np.median([each.obsm['spatial'][:,0].max()-each.obsm['spatial'][:,0].min()+1 for each in rescaled] )
    rangeY=np.median([each.obsm['spatial'][:,1].max()-each.obsm['spatial'][:,1].min()+1 for each in rescaled] )
    tissuelength=min(rangeX, rangeY)

    # use padding to center the spots 
    centerpadlengthXs=[]
    centerpadlengthYs=[]
    for each in range(len(rescaled)):
        htif,wtif,ncolor=rescaled[each].uns[spatial_key][library_key]['images'][res].shape
        xy_indices=rescaled[each].obsm['spatial'].copy()
        scalehigh=rescaled[each].uns[spatial_key][library_key]['scalefactors']['tissue_'+str(res)+'_scalef']
        bestpadlengthX= np.mean([xy_indices[:,0].min(), wtif/scalehigh-xy_indices[:,0].max()])#padding length
        bestpadlengthY= np.mean([xy_indices[:,1].min(), htif/scalehigh-xy_indices[:,1].max()]) #padding length
        centerpadlengthXs.append(bestpadlengthX)
        centerpadlengthYs.append(bestpadlengthY)
    centerpadlengthX=np.max(centerpadlengthXs)
    centerpadlengthY=np.max(centerpadlengthYs)
    for each in range(len(rescaled)):
        rescaled[each]=padAdataByHires(rescaled[each],centerpadlengthX,centerpadlengthY, finalhires=round(finalhires*1.25))
    
    # step 4
    padlengthXs=[]
    padlengthYs=[]
    for each in range(len(rescaled)):
        htif,wtif,ncolor=rescaled[each].uns[spatial_key][library_key]['images'][res].shape
        scalehigh=rescaled[each].uns[spatial_key][library_key]['scalefactors']['tissue_'+str(res)+'_scalef']
        xy_indices=rescaled[each].obsm['spatial'].copy()
        x_min_bytissue=np.where(rescaled[each].uns[spatial_key][library_key]['images']['hires'].sum(axis=2)>0)[1].min()/scalehigh
        x_max_bytissue=np.where(rescaled[each].uns[spatial_key][library_key]['images']['hires'].sum(axis=2)>0)[1].max()/scalehigh
        y_min_bytissue=np.where(rescaled[each].uns[spatial_key][library_key]['images']['hires'].sum(axis=2)>0)[0].min()/scalehigh
        y_max_bytissue=np.where(rescaled[each].uns[spatial_key][library_key]['images']['hires'].sum(axis=2)>0)[0].max()/scalehigh
        bestpadlengthX=max(max(round(tissuelength*ratio),xy_indices[:,0].min()-x_min_bytissue+5), (x_max_bytissue-xy_indices[:,0].max()+5))
        bestpadlengthY=max(max(round(tissuelength*ratio),xy_indices[:,1].min()-y_min_bytissue+5), (y_max_bytissue-xy_indices[:,1].max()+5)) #padding length
        padlengthXs.append(bestpadlengthX)
        padlengthYs.append(bestpadlengthY)
    bestpadlengthX=np.max(padlengthXs)
    bestpadlengthY=np.max(padlengthYs)
    
    shrinks=[]
    for each in range(len(rescaled)):
        scalehigh=rescaled[each].uns[spatial_key][library_key]['scalefactors']['tissue_'+str(res)+'_scalef']
        xy_indices=rescaled[each].obsm['spatial'].copy()
        newMinX=xy_indices[:,0].min()-bestpadlengthX # col?
        newMinY=xy_indices[:,1].min()-bestpadlengthY # row?
        newMaxX=xy_indices[:,0].max()+bestpadlengthX
        newMaxY=xy_indices[:,1].max()+bestpadlengthY
        shrink=min(1,(finalhires-1)*1.0/(round(newMaxX*scalehigh)-round(newMinX*scalehigh)))
        shrinks.append(shrink)
    bestshrink=np.min(shrinks)
    bestpadlengthX=bestpadlengthX*bestshrink
    bestpadlengthY=bestpadlengthY*bestshrink
    
    rescaled2=[]
    for each in range(len(rescaled)):
        tmp=rescaled[each].copy()
        img=Image.fromarray((rescaled[each].uns[spatial_key][library_key]['images'][res]*255).astype(np.uint8))
        img=img.resize([round(dim*bestshrink) for dim in img.size])
        imgarray=np.array(img)/255.0
        tmp.uns[spatial_key][library_key]['images'][res]=imgarray
        tmp.obsm['spatial']=bestshrink*tmp.obsm['spatial']
        rescaled2.append(tmp)
    rescaled=rescaled2
    
    #print([bestpadlengthX,bestpadlengthY])
    #print([each.obsm['spatial'][:,0].min() for each in rescaled])
    #print([each.obsm['spatial'][:,1].min() for each in rescaled])
    
    for each in range(len(rescaled)):
        rescaled[each]=cropAdataByHires(rescaled[each],bestpadlengthX,bestpadlengthY, finalhires=finalhires)
    return rescaled

def cropAdataByHiresWithMultiImg(adata,bestpadlengthX,bestpadlengthY, reskeys,finalhires=None):
    adata_cp=adata.copy()
    spatial_key=list(adata_cp.uns.keys())[0]
    library_key=list(adata_cp.uns[spatial_key].keys())[0]
    scalehigh=adata_cp.uns[spatial_key][library_key]['scalefactors']['tissue_'+str('hires')+'_scalef']        
    himgX=adata_cp.uns[spatial_key][library_key]['images']['hires'].shape[0]
    himgY=adata_cp.uns[spatial_key][library_key]['images']['hires'].shape[1]
    limgX=adata_cp.uns[spatial_key][library_key]['images']['hires'].shape[0]
    limgY=adata_cp.uns[spatial_key][library_key]['images']['hires'].shape[1]
    
    xy_indices=adata_cp.obsm['spatial'].copy()
    x0=xy_indices[:,0].min()
    y0=xy_indices[:,1].min()
    xM=xy_indices[:,0].max()
    yM=xy_indices[:,1].max()        

    newMinX=x0-bestpadlengthX # col?
    newMinY=y0-bestpadlengthY # row?
    newMaxX=xM+bestpadlengthX
    newMaxY=yM+bestpadlengthY
    
    if finalhires is not None:
        #if round(newMaxX*scalehigh)+1>round(newMinX*scalehigh)+finalhires:
        if round(xM*scalehigh)+1>round(x0*scalehigh)+finalhires*1.02 or round(yM*scalehigh)+1>round(y0*scalehigh)+finalhires*1.02:
            print("ERROR: image width is too large to fit finalhires without cropping out tissue")
            return None

    #shift spot coords to match newly cropped img
    xy_indices[:,0]=xy_indices[:,0]-newMinX # = padlength
    xy_indices[:,1]=xy_indices[:,1]-newMinY
    hnewMinX=round(newMinX*scalehigh)+0 #PIL Image cropping is 0-index and end-exclusive
    hnewMinY=round(newMinY*scalehigh)+0
    hnewMaxX=round(newMaxX*scalehigh)+1
    hnewMaxY=round(newMaxY*scalehigh)+1
    if finalhires is not None:
        #round(newMaxX*scalehigh)+1-round(newMinX*scalehigh)=finalhires
        hnewMaxX=round(newMinX*scalehigh)+finalhires
        hnewMaxY=round(newMinY*scalehigh)+finalhires
    for res in reskeys:
     hcropped=Image.fromarray((adata_cp.uns[spatial_key][library_key]['images'][res]* 255).astype(np.uint8)).convert('RGB').crop((hnewMinX,hnewMinY,hnewMaxX,hnewMaxY))
     adata_cp.uns[spatial_key][library_key]['images'][res]=(np.asarray(hcropped)/255).astype(np.float32)

    if 'tissue_'+str('lowres')+'_scalef' in list(adata_cp.uns[spatial_key][library_key]['scalefactors'].keys()):
        scalelow=adata_cp.uns[spatial_key][library_key]['scalefactors']['tissue_'+str('lowres')+'_scalef']
        lnewMinX=round(newMinX*scalelow)+0 #PIL Image cropping is 0-index and end-exclusive
        lnewMaxX=round(newMaxX*scalelow)+1
        lnewMinY=round(newMinY*scalelow)+0
        lnewMaxY=round(newMaxY*scalelow)+1
        if finalhires is not None:
            lnewMaxX=round(newMinX*scalelow)+round(finalhires*scalelow/scalehigh)
            lnewMaxY=round(newMinY*scalelow)+round(finalhires*scalelow/scalehigh)
        lcropped=Image.fromarray((adata_cp.uns[spatial_key][library_key]['images']['lowres']* 255).astype(np.uint8)).convert('RGB').crop((lnewMinX,lnewMinY,lnewMaxX,lnewMaxY))
        adata_cp.uns[spatial_key][library_key]['images']['lowres']=(np.asarray(lcropped)/255).astype(np.float32)

    adata_cp.obs=adata_cp.obs.loc[adata_cp.obs['in_tissue']==1] #new obs
    adata_cp.obsm['spatial']=xy_indices
    return adata_cp

def padAdataByHiresWithMultiImg(adata,bestpadlengthX,bestpadlengthY, reskeys,finalhires=None):
    adata_cp=adata.copy()
    spatial_key=list(adata_cp.uns.keys())[0]
    library_key=list(adata_cp.uns[spatial_key].keys())[0]
    scalehigh=adata_cp.uns[spatial_key][library_key]['scalefactors']['tissue_'+str('hires')+'_scalef']   

    xy_indices=adata_cp.obsm['spatial'].copy()
    padX4FinalHires=0
    padY4FinalHires=0
    if finalhires is not None:
            padX4FinalHires=(finalhires/scalehigh-(xy_indices[:,0].max()-xy_indices[:,0].min()+1))/2
            padY4FinalHires=(finalhires/scalehigh-(xy_indices[:,1].max()-xy_indices[:,1].min()+1))/2
    bestpadlengthX=max(padX4FinalHires, bestpadlengthX)#padding length
    bestpadlengthY=max(padY4FinalHires, bestpadlengthY) #padding length
        
    x0=xy_indices[:,0].min()
    y0=xy_indices[:,1].min()
    xM=xy_indices[:,0].max()
    yM=xy_indices[:,1].max()        

    newMinX=x0-bestpadlengthX # col?
    newMinY=y0-bestpadlengthY # row?
    
    #shift spot coords to match newly cropped img
    xy_indices[:,0]=xy_indices[:,0]-newMinX
    xy_indices[:,1]=xy_indices[:,1]-newMinY
    
    himgMinX=round(-(newMinX)*scalehigh)+0
    himgMinY=round(-(newMinY)*scalehigh)+0
    hnew_width = round((xy_indices[:,0].max()-xy_indices[:,0].min()+1+2*bestpadlengthX)*scalehigh)
    hnew_height = round((xy_indices[:,1].max()-xy_indices[:,1].min()+1+2*bestpadlengthY)*scalehigh)
    if finalhires is not None:
        hnew_width=max(hnew_width,finalhires)
        hnew_height=max(hnew_height,finalhires)

    for res in reskeys:
     hpadded=Image.new("RGB", (hnew_width, hnew_height), (0, 0, 0))
     orgimg=Image.fromarray((adata_cp.uns[spatial_key][library_key]['images'][res] * 255).astype(np.uint8)).convert('RGB')
     hpadded.paste(orgimg,(himgMinX,himgMinY))
     adata_cp.uns[spatial_key][library_key]['images'][res]=(np.asarray(hpadded)/255).astype(np.float32)

    if 'tissue_'+str('lowres')+'_scalef' in list(adata_cp.uns[spatial_key][library_key]['scalefactors'].keys()):
        scalelow=adata_cp.uns[spatial_key][library_key]['scalefactors']['tissue_'+str('lowres')+'_scalef']
        limgMinX=round(-(newMinX)*scalelow)+0
        limgMinY=round(-(newMinY)*scalelow)+0  
        lnew_width = round((xy_indices[:,0].max()-xy_indices[:,0].min()+1+2*bestpadlengthX)*scalelow)
        lnew_height = round((xy_indices[:,1].max()-xy_indices[:,1].min()+1+2*bestpadlengthY)*scalelow)
        if finalhires is not None:
            lnew_width=max(lnew_width,round(finalhires*scalelow/scalehigh))
            lnew_height=max(lnew_height,round(finalhires*scalelow/scalehigh))
        lpadded=Image.new("RGB", (lnew_width, lnew_height), (0, 0, 0))
        orgimg=Image.fromarray((adata_cp.uns[spatial_key][library_key]['images']['lowres'] * 255).astype(np.uint8)).convert('RGB')
        lpadded.paste(orgimg,(limgMinX,limgMinY))
        adata_cp.uns[spatial_key][library_key]['images']['lowres']=(np.asarray(lpadded)/255).astype(np.float32)

    adata_cp.obs=adata_cp.obs.loc[adata_cp.obs['in_tissue']==1] #new obs
    adata_cp.obsm['spatial']=xy_indices
    return adata_cp

def afTransformWithMultiImg(im_mov, im_fix, mask_mov, mask_fix,otherimgs, spatial_mov,max_reg_dim=512,defaultvalue=0, search_factor=5 ,use_principal_axis=True, type_of_transform='Affine'):
    
    # make a "small" size for faster af
    im_mov_r, im_fix_r, mask_mov_r, mask_fix_r = map(lambda x, resample: resize_to_max_dim_pil(x,
                                                                                                max_dim=max_reg_dim,
                                                                                                resample=resample),
                                                        (im_mov, im_fix, mask_mov, mask_fix),
                                                        (Image.Resampling.BILINEAR,
                                                        Image.Resampling.BILINEAR, #
                                                        Image.Resampling.NEAREST,
                                                        Image.Resampling.NEAREST))
#    otherimgs_r=[resize_to_max_dim_pil(x,max_dim=max_reg_dim,resample=Image.BILINEAR) for x in otherimgs]
    
    
    # perform affine registration; omit itk for now.
    if search_factor>=20:
         #print(' initializer is on')
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

    # upscale the affine transform
    large_aff_tfm = ants.read_transform(aff_tfm_info['fwdtransforms'][0]) # assume only one matrix out.
    large_aff_tfm = scale_affine(trf=large_aff_tfm,
                                        orig_shape_2d=im_mov_r.size[::-1],
                                        new_shape_2d=im_mov.size[::-1])
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
    otherimgs_a=[apply_itk_trf_image(input=np.asarray(x).astype(float),
                                     trf=large_aff_tfm_itk,
                                     interpolator=sitk.sitkLinear,
                                     defaultPixelValue=float(defaultvalue),
                                     outputPixelType=sitk.sitkUnknown,
                                     useNearestNeighborExtrapolator=True
                                    ) for x in otherimgs]
    
    inv_large_aff_tfm_itk=large_aff_tfm_itk.GetInverse()
    # need to apply inverse transform to mimic image's resample.
    spatial_mov_a = register_spatial_with_itk_points(
                    spatial_data=spatial_mov,
                    inverse_itk_trf=inv_large_aff_tfm_itk, 
                    spatial_data_indexing='xy')
    return im_mov_a, mask_mov_a, spatial_mov_a, otherimgs_a


def scaleAdataWithMultiImg(adataset,  ratio=0.07, finalhires=512, coarse=0, onlyCropPad=False, search_factor=20, use_principal_axis=False,type_of_transform='Affine',mask_fn=None):
    rescaled=[]
    spatial_key=list(adataset[0].uns.keys())[0]
    library_key=list(adataset[0].uns[spatial_key].keys())[0]
    map2keys={}
    if not onlyCropPad:
     af=[]
     for each in range(len(adataset)):
        
        reskeys=list(adataset[each].uns[spatial_key][library_key]['images'].keys())
        reskeys=np.unique([re.sub("_mask", "",x) for x in reskeys]).tolist()        
        tmp=adataset[each].copy()
        spatial_mov=tmp.uns[spatial_key].values()
        movd_spatial_nm = next(iter(tmp.uns[spatial_key].keys()))

        for res in reskeys:
         im_mov=Image.fromarray((tmp.uns[spatial_key][movd_spatial_nm]['images'][res] * 255).astype(np.uint8)).convert('RGB')         
         [im_mov, mask_mov]=prep4AF(im_mov, mask_mov=None, mask_fn=mask_fn)
         tmp.uns['spatial'][movd_spatial_nm]['images'][res]=im_mov
         if res == 'hires':
           tmp.uns['spatial'][movd_spatial_nm]['images']['%s_mask' % res]=mask_mov
        
        scalefactor_mov = tmp.uns[spatial_key][movd_spatial_nm]['scalefactors']['tissue_%s_scalef' % 'hires']
        spatial_mov_scaled = tmp.obsm['spatial'] * scalefactor_mov
        tmp.obsm['spatial'] =spatial_mov_scaled
        af.append(tmp)
        
     rest=set(range(len(adataset))).difference(set([coarse]))
     im_fix=af[coarse].uns[spatial_key][next(iter(af[coarse].uns[spatial_key].keys()))]['images']['hires']
     mask_fix=af[coarse].uns[spatial_key][next(iter(af[coarse].uns[spatial_key].keys()))]['images']['%s_mask' % 'hires']
     for each in rest:
        reskeys=list(af[each].uns[spatial_key][library_key]['images'].keys())
        reskeys=np.unique([re.sub("_mask", "",x) for x in reskeys]).tolist()        
        
        movd_spatial_nm=next(iter(af[each].uns[spatial_key].keys()))
        im_mov=af[each].uns[spatial_key][movd_spatial_nm]['images']['hires']
        mask_mov=af[each].uns[spatial_key][movd_spatial_nm]['images']['%s_mask' % 'hires']
        spatial_mov=af[each].obsm['spatial']
        otherimgkeys=list(set(reskeys).difference(set(['hires'])))
        otherimgs=[af[each].uns['spatial'][movd_spatial_nm]['images'][x] for x in otherimgkeys]
        [im_mov_a,mask_mov_a, spatial_mov_a, otherimgs_a]=afTransformWithMultiImg(im_mov, im_fix, mask_mov, mask_fix,otherimgs,spatial_mov,max_reg_dim=256, search_factor=search_factor,
                                                       use_principal_axis=use_principal_axis,type_of_transform=type_of_transform)
        af[each].uns['spatial'][movd_spatial_nm]['images']['hires']=im_mov_a
        af[each].obsm['spatial']=spatial_mov_a
        for s in range(len(otherimgkeys)):
            af[each].uns['spatial'][movd_spatial_nm]['images'][otherimgkeys[s]]=otherimgs_a[s]

     for each in range(len(af)):
        movd_spatial_nm=next(iter(af[each].uns[spatial_key].keys()))
        af[each].uns[spatial_key][movd_spatial_nm]['scalefactors']['tissue_%s_scalef' % 'hires']=1
        af[each].uns[spatial_key][movd_spatial_nm]['scalefactors']['tissue_%s_scalef' % 'lowres']=1
        af[each].obsm['spatial']= af[each].obsm['spatial'] #/ scalefactor_mov
        reskeys=list(af[each].uns[spatial_key][library_key]['images'].keys())
        reskeys=np.unique([re.sub("_mask", "",x) for x in reskeys]).tolist()        
        for res in reskeys:
             af[each].uns['spatial'][movd_spatial_nm]['images'][res] = np.asarray(af[each].uns['spatial'][movd_spatial_nm]['images'][res]) / 255
             if res=='hires':
                del af[each].uns[spatial_key][movd_spatial_nm]['images']['%s_mask' % res] #remove  mask to be sure 
    
    #first scaling
     rangeX=np.median([each.obsm['spatial'][:,0].max()-each.obsm['spatial'][:,0].min()+1 for each in af] )
     rangeY=np.median([each.obsm['spatial'][:,1].max()-each.obsm['spatial'][:,1].min()+1 for each in af] )
     scalings=[min(rangeX/(each.obsm['spatial'][:,0].max()-each.obsm['spatial'][:,0].min()+1),rangeY/(each.obsm['spatial'][:,1].max()-each.obsm['spatial'][:,1].min()+1)) for each in af]
    
     for each in range(len(af)):
        tmp=af[each].copy()
        movd_spatial_nm = next(iter(tmp.uns[spatial_key].keys()))
        reskeys=list(af[each].uns[spatial_key][library_key]['images'].keys())
        reskeys=np.unique([re.sub("_mask", "",x) for x in reskeys]).tolist()        
        for res in reskeys:
         img=Image.fromarray((af[each].uns[spatial_key][movd_spatial_nm]['images'][res]*255).astype(np.uint8))
         img=img.resize([round(dim*scalings[each]) for dim in img.size])
         imgarray=np.array(img)/255.0
         tmp.uns[spatial_key][library_key]['images'][res]=imgarray

        tmp.obsm['spatial']=scalings[each]*tmp.obsm['spatial']
        rescaled.append(tmp)
    else: # if only need to crop and pad to reach final resolution
     rescaled=[x.copy() for x in adataset]   

    # update ranges and tissuelength with rescaled 
    rangeX=np.median([each.obsm['spatial'][:,0].max()-each.obsm['spatial'][:,0].min()+1 for each in rescaled] )
    rangeY=np.median([each.obsm['spatial'][:,1].max()-each.obsm['spatial'][:,1].min()+1 for each in rescaled] )
    tissuelength=min(rangeX, rangeY)

    # use padding to center the spots 
    centerpadlengthXs=[]
    centerpadlengthYs=[]
    for each in range(len(rescaled)):
        xy_indices=rescaled[each].obsm['spatial'].copy()
        
        htif,wtif,ncolor=rescaled[each].uns[spatial_key][library_key]['images']['hires'].shape

        scalehigh=rescaled[each].uns[spatial_key][library_key]['scalefactors']['tissue_'+str('hires')+'_scalef']
        bestpadlengthX= np.mean([xy_indices[:,0].min(), wtif/scalehigh-xy_indices[:,0].max()])#padding length
        bestpadlengthY= np.mean([xy_indices[:,1].min(), htif/scalehigh-xy_indices[:,1].max()]) #padding length
        centerpadlengthXs.append(bestpadlengthX)
        centerpadlengthYs.append(bestpadlengthY)
    centerpadlengthX=np.max(centerpadlengthXs)
    centerpadlengthY=np.max(centerpadlengthYs)
    print([centerpadlengthX, centerpadlengthY])
    for each in range(len(rescaled)):
        reskeys=list(rescaled[each].uns[spatial_key][library_key]['images'].keys())
        reskeys=np.unique([re.sub("_mask", "",x) for x in reskeys]).tolist()        
        rescaled[each]=padAdataByHiresWithMultiImg(rescaled[each],centerpadlengthX,centerpadlengthY,reskeys=reskeys, finalhires=round(finalhires*1.25))
    
    # step 4
    padlengthXs=[]
    padlengthYs=[]
    for each in range(len(rescaled)):
        htif,wtif,ncolor=rescaled[each].uns[spatial_key][library_key]['images']['hires'].shape
        scalehigh=rescaled[each].uns[spatial_key][library_key]['scalefactors']['tissue_'+str('hires')+'_scalef']
        xy_indices=rescaled[each].obsm['spatial'].copy()
        x_min_bytissue=np.where(rescaled[each].uns[spatial_key][library_key]['images']['hires'].sum(axis=2)>0)[1].min()/scalehigh
        x_max_bytissue=np.where(rescaled[each].uns[spatial_key][library_key]['images']['hires'].sum(axis=2)>0)[1].max()/scalehigh
        y_min_bytissue=np.where(rescaled[each].uns[spatial_key][library_key]['images']['hires'].sum(axis=2)>0)[0].min()/scalehigh
        y_max_bytissue=np.where(rescaled[each].uns[spatial_key][library_key]['images']['hires'].sum(axis=2)>0)[0].max()/scalehigh
        bestpadlengthX=max(max(round(tissuelength*ratio),xy_indices[:,0].min()-x_min_bytissue+5), (x_max_bytissue-xy_indices[:,0].max()+5))
        bestpadlengthY=max(max(round(tissuelength*ratio),xy_indices[:,1].min()-y_min_bytissue+5), (y_max_bytissue-xy_indices[:,1].max()+5)) #padding length
        padlengthXs.append(bestpadlengthX)
        padlengthYs.append(bestpadlengthY)
    bestpadlengthX=np.max(padlengthXs)
    bestpadlengthY=np.max(padlengthYs)
    
    shrinks=[]
    for each in range(len(rescaled)):
        scalehigh=rescaled[each].uns[spatial_key][library_key]['scalefactors']['tissue_'+str("hires")+'_scalef']
        xy_indices=rescaled[each].obsm['spatial'].copy()
        newMinX=xy_indices[:,0].min()-bestpadlengthX # col?
        newMinY=xy_indices[:,1].min()-bestpadlengthY # row?
        newMaxX=xy_indices[:,0].max()+bestpadlengthX
        newMaxY=xy_indices[:,1].max()+bestpadlengthY
        shrink=min(1,(finalhires-1)*1.0/(round(newMaxX*scalehigh)-round(newMinX*scalehigh)))
        shrinks.append(shrink)
    bestshrink=np.min(shrinks)
    bestpadlengthX=bestpadlengthX*bestshrink
    bestpadlengthY=bestpadlengthY*bestshrink
    
    rescaled2=[]
    for each in range(len(rescaled)):
        tmp=rescaled[each].copy()
        reskeys=list(rescaled[each].uns[spatial_key][library_key]['images'].keys())
        reskeys=np.unique([re.sub("_mask", "",x) for x in reskeys]).tolist()        
        for res in reskeys:         
         img=Image.fromarray((rescaled[each].uns[spatial_key][library_key]['images'][res]*255).astype(np.uint8))
         img=img.resize([round(dim*bestshrink) for dim in img.size])
         imgarray=np.array(img)/255.0
         tmp.uns[spatial_key][library_key]['images'][res]=imgarray
         #print([res,imgarray.shape[0],imgarray.shape[1]]) 

        tmp.obsm['spatial']=bestshrink*tmp.obsm['spatial']
        rescaled2.append(tmp)
    rescaled=rescaled2
    
    #print([bestpadlengthX,bestpadlengthY])
    #print([each.obsm['spatial'][:,0].min() for each in rescaled])
    #print([each.obsm['spatial'][:,1].min() for each in rescaled])
    
    for each in range(len(rescaled)):
        reskeys=list(rescaled[each].uns[spatial_key][library_key]['images'].keys())
        reskeys=np.unique([re.sub("_mask", "",x) for x in reskeys]).tolist()        
        rescaled[each]=cropAdataByHiresWithMultiImg(rescaled[each],bestpadlengthX,bestpadlengthY, reskeys=reskeys,finalhires=finalhires)
    return rescaled


def applyDffToOthers(inadata, otherres, large_aff_tfm_itk, large_global_deff):
    #large_aff_tfm_itk=outputs[1]['affine']['deff']
    #large_global_deff=outputs[1]['dense']['deff']
    defaultvalue=0
    adata=inadata.copy()
    for res in otherres:
        x=adata.uns['spatial']['lib']['images'][res]
        x_a=apply_itk_trf_image(input=np.asarray(x).astype(float),
                                     trf=large_aff_tfm_itk,
                                     interpolator=sitk.sitkLinear,
                                     defaultPixelValue=float(defaultvalue),
                                     outputPixelType=sitk.sitkUnknown,
                                     useNearestNeighborExtrapolator=True
                                    ) 
        x_d=vxm.layers.SpatialTransformer(interp_method='linear',indexing='ij')([tf.expand_dims(x_a.astype(float), axis=0),tf.expand_dims(large_global_deff.astype(float), axis=0)])[0].numpy()
        #x_d=x_d/255.0
        adata.uns['spatial']['lib']['images'][res]=x_d
    return adata


def applyDffToAll(inadata, large_aff_tfm_itk, large_global_deff,large_global_inv_deff):
    defaultvalue=0
    adata=inadata.copy()
    allres=list(adata.uns['spatial']['lib']['images'].keys())
    for res in allres:
        x=adata.uns['spatial']['lib']['images'][res]
        x_a=apply_itk_trf_image(input=np.asarray(x).astype(float),
                                     trf=large_aff_tfm_itk,
                                     interpolator=sitk.sitkLinear,
                                     defaultPixelValue=float(defaultvalue),
                                     outputPixelType=sitk.sitkUnknown,
                                     useNearestNeighborExtrapolator=True
                                    ) 
        x_d=vxm.layers.SpatialTransformer(interp_method='linear',indexing='ij')([tf.expand_dims(x_a.astype(float), axis=0),tf.expand_dims(large_global_deff.astype(float), axis=0)])[0].numpy()
        #x_d=x_d/255.0
        adata.uns['spatial']['lib']['images'][res]=x_d
    spatial_mov_a = register_spatial_with_itk_points(
                    spatial_data=adata.obsm['spatial'],
                    inverse_itk_trf=large_aff_tfm_itk.GetInverse(), 
                    spatial_data_indexing='xy')
    spatial_mov_d = register_spatial_with_def_field_points(
                    spatial_data=spatial_mov_a,
                    inverse_def_field=large_global_inv_deff,
                    spatial_data_indexing='xy',
                    inverse_def_field_indexing='ij')
    adata.obsm['spatial']=spatial_mov_d
    return adata
    

def compileADataWithDummySpots(imghires,otherimgs,initscalefactor=1,otherlabels=None):
    #initscalefactor: imghirs size/original full resolution image size
    var=pd.DataFrame({'gene_ids':['g1','g2','g3']})
    var['feature_types']='Gene Expression'
    var['genome']='mm10'
    var=var.set_index('gene_ids',drop=False)
    var.index.name = None
    
    tissuecoords=(np.argwhere(imghires[:,:,0]>0)) #[height, width],[y, x]
    tmp=tissuecoords.copy()
    #tmp=(np.round(tmp/initscalefactor,0)).astype(np.int64)
    tmp[:,0]=tissuecoords[:,1]
    tmp[:,1]=tissuecoords[:,0]
    tissuecoords=tmp
    
    sel=[np.where(tissuecoords[:,0]==tissuecoords[:,0].min())[0][0],
        np.where(tissuecoords[:,0]==tissuecoords[:,0].max())[0][0],
        np.where(tissuecoords[:,1]==tissuecoords[:,1].min())[0][0],
        np.where(tissuecoords[:,1]==tissuecoords[:,1].max())[0][0]]
    
    obsm={'spatial': (np.round(tissuecoords[sel,:]/initscalefactor,0)).astype(np.int64)}
    # obsm={'spatial':np.array([[tissuecoords[:,0].min(),tissuecoords[:,1].min()],
    #                       [tissuecoords[:,0].min(),tissuecoords[:,1].max()],
    #                       [tissuecoords[:,0].max(),tissuecoords[:,1].min()],
    #                       [tissuecoords[:,0].max(),tissuecoords[:,1].max()]])}
    obs=pd.DataFrame(obsm['spatial'],index=['sp1','sp2','sp3','sp4'],columns=['imagecol','imagerow'])
    obs['in_tissue']=1
    
    lowres=imghires.copy()
    img_dict={'hires':imghires,'lowres':lowres}
    for each in range(len(otherimgs)):
        if otherlabels is not None:
            img_dict[otherlabels[each]]=otherimgs[each]
        else:
            img_dict['layer'+str((each+1))]=otherimgs[each]
    uns={'spatial':{'lib':{'images':img_dict,'scalefactors':{'tissue_hires_scalef':initscalefactor,
                                                             'tissue_lowres_scalef':initscalefactor*lowres.shape[0]*1.0/imghires.shape[0],
                                                             'spot_diameter_fullres':12}}}}

    X=np.ones((obsm['spatial'].shape[0],var.shape[0]))

    ihc=ad.AnnData(X, obs=obs, var=var,obsm=obsm, uns=uns)
    return ihc
      
def resizeADataImgWithMultiImg(adata, max_dim):
    spatial_key=list(adata.uns.keys())[0]
    lib_key=list(adata.uns[spatial_key].keys())[0]
    newdata=adata.copy()
    for res in list(adata.uns[spatial_key][lib_key]['images'].keys()):
        imgarray=adata.uns[spatial_key][lib_key]['images'][res]
        
        scalefactor=adata.uns[spatial_key][lib_key]['scalefactors']['tissue_hires_scalef']
        imgtif=Image.fromarray((imgarray*255).astype(np.uint8))
        img4k=resize_to_max_dim_pil(imgtif,max_dim=max_dim)
        img4karray=np.asarray(img4k,dtype='float32')/255
        scale4k=img4karray.shape[1]/imgtif.size[0]*scalefactor
        newdata.uns[spatial_key][lib_key]['images'][res]=img4karray

    
    newdata.uns[spatial_key][lib_key]['scalefactors']['tissue_hires_scalef']=scale4k
    return newdata


def cleanEdge(inarray, left, right, top, bottom):
    imgarray=inarray.copy()
    edgecolor=imgarray[imgarray.shape[0]-int(imgarray.shape[0]*0.02),imgarray.shape[1]-int(imgarray.shape[1]*0.02),:]
    imgarray[:,0:left,0]=edgecolor[0]
    imgarray[:,0:left,1]=edgecolor[1]
    imgarray[:,0:left,2]=edgecolor[2]
    
    imgarray[:,right:imgarray.shape[1],0]=edgecolor[0]
    imgarray[:,right:imgarray.shape[1],1]=edgecolor[1]
    imgarray[:,right:imgarray.shape[1],2]=edgecolor[2]
    imgarray[0:top,:,0]=edgecolor[0]
    imgarray[0:top,:,1]=edgecolor[1]
    imgarray[0:top,:,2]=edgecolor[2]
    imgarray[bottom:imgarray.shape[0],:,0]=edgecolor[0]
    imgarray[bottom:imgarray.shape[0],:,1]=edgecolor[1]
    imgarray[bottom:imgarray.shape[0],:,2]=edgecolor[2]
    return imgarray

def cleanArea(inarray, left, right, top, bottom):
    imgarray=inarray.copy()
    edgecolor=imgarray[imgarray.shape[0]-int(imgarray.shape[0]*0.02),imgarray.shape[1]-int(imgarray.shape[1]*0.02),:]
    
    X2D,Y2D = np.meshgrid(np.arange(left,right),np.arange(top,bottom))
    out = np.column_stack((Y2D.ravel(),X2D.ravel()))

    imgarray[out[:,0],out[:,1],0]=edgecolor[0]
    imgarray[out[:,0],out[:,1],1]=edgecolor[1]
    imgarray[out[:,0],out[:,1],2]=edgecolor[2]
    return imgarray

def scale_spatial_coords2(X, ref_min=None, ref_max=None, max_val=10.0):
    if ref_min is None:
        ref_min = X.min(0)
    X = X - ref_min
    if ref_max is None:
        ref_max = X.max(0)
    X = X / ref_max
    return X * max_val

def createGrid(xDim=512, yDim=512):
 grid=[]
 for x in range(xDim): #adata_warped.uns[spatial_key][library_key]['images']['hires'].shape[1]
    for y in range(yDim):
        grid.append([x,y])
 grid=np.array(grid)
 return grid


def deffPixel(rs, pixel_org):
    pixel_a=register_spatial_with_itk_points(pixel_org,
                                         rs[1]['affine']['inv_deff'],
                                         spatial_data_indexing='ij')
    pixel_d=register_spatial_with_def_field_points(
                    spatial_data=pixel_a,
                    inverse_def_field=rs[1]['dense']['inv_deff'],
                    spatial_data_indexing='xy',
                    inverse_def_field_indexing='ij')    
    return pixel_d


def iterModel(adata_unwarped, adata_warped, model_sm, resolution=512, maxRound=4, maxratio=0.01, only1round=False, search_factor=5):
  spatial_key=list(adata_unwarped.uns.keys())[0]
  library_id=list(adata_unwarped.uns[spatial_key].keys())[0]
    
  pixel_org=createGrid(xDim=adata_warped.uns[spatial_key][library_id]['images']['hires'].shape[1], yDim=adata_warped.uns[spatial_key][library_id]['images']['hires'].shape[0])

  new_img_iter1=[]
  new_spatial_iter1=[]
  new_mse_iter1=[]
  new_mse=[]
  new_spatial=[]
  new_img=[]
    #first round  
  registration_data_points = sc_register(adata_mov=adata_warped,
                     adata_fix=adata_unwarped,
                     spatial_strategy='points',
                     spatial_target='hires',
                     max_reg_dim=resolution,
                     mode='dense',
                     model=model_sm,
                     dense_moving_img_name='sm_semi_base_moving_img',
                     dense_fixed_img_name='sm_semi_base_fixed_img',
                     dense_deff_name='def_output',
                     dense_inv_deff_name='inv_def_output',
                     search_factor=search_factor,
                     use_principal_axis=False)
  
  if only1round:
    return registration_data_points
  pixel_d=deffPixel(registration_data_points, pixel_org)
    
  fit=scale_spatial_coords2(registration_data_points[0].obsm['spatial'],ref_min=adata_warped.obsm['spatial'].min(0),
                          ref_max=(adata_warped.obsm['spatial']-adata_warped.obsm['spatial'].min(0)).max(0))
    
  previous=registration_data_points[0].copy()
  previousfit=fit
  
  #2nd round
    
  registration_data_points = sc_register(adata_mov=previous,
                     adata_fix=adata_unwarped,
                     spatial_strategy='points',
                     max_reg_dim=resolution,
                     mode='dense',
                     model=model_sm,
                     dense_moving_img_name='sm_semi_base_moving_img',
                     dense_fixed_img_name='sm_semi_base_fixed_img',
                     dense_deff_name='def_output',
                     dense_inv_deff_name='inv_def_output',
                     spot_diameter_unscaled=1,
                     dense_normalize=True)
  pixel_d=deffPixel(registration_data_points, pixel_d)

  fit=scale_spatial_coords2(registration_data_points[0].obsm['spatial'],ref_min=adata_warped.obsm['spatial'].min(0),
                          ref_max=(adata_warped.obsm['spatial']-adata_warped.obsm['spatial'].min(0)).max(0))
  first_delta_fit=(((fit[:,0] - previousfit[:,0])** 2+(fit[:,1] - previousfit[:,1])** 2).mean())
  round=1
  while round <=maxRound:
   registration_data_points = sc_register(adata_mov=previous,
                     adata_fix=adata_unwarped,
                     spatial_strategy='points',
                     max_reg_dim=resolution,
                     mode='dense',
                     model=model_sm,
                     dense_moving_img_name='sm_semi_base_moving_img',
                     dense_fixed_img_name='sm_semi_base_fixed_img',
                     dense_deff_name='def_output',
                     dense_inv_deff_name='inv_def_output',
                     spot_diameter_unscaled=1,
                     dense_normalize=True)
   pixel_d=deffPixel(registration_data_points, pixel_d)

   round=round+1
   fit=scale_spatial_coords2(registration_data_points[0].obsm['spatial'],ref_min=adata_warped.obsm['spatial'].min(0),
                          ref_max=(adata_warped.obsm['spatial']-adata_warped.obsm['spatial'].min(0)).max(0))
   delta_fit=(((fit[:,0] - previousfit[:,0])** 2+(fit[:,1] - previousfit[:,1])** 2).mean())
   dd_fit=abs(delta_fit-first_delta_fit) 
   if (dd_fit/first_delta_fit)<maxratio:
        break
   first_delta_fit= delta_fit
   previous=registration_data_points[0].copy()
   previousfit= fit
   pixel_final=deffPixel(registration_data_points, pixel_d)
   registration_data_points[1]['pixel']=pixel_d
    
  return registration_data_points


from sklearn.metrics import adjusted_rand_score 
def metric_ltari (refspot, reflbl, queryspot, querylbl, knn=5):
    #queryspot: spots whose label to be inferred from refspot & reflbl
    #querylbl: ground truth of labels for queryspots
    node_dict = dict(zip(range(refspot.shape[0]), reflbl.index))
    D=distance_matrix(queryspot,refspot)
    idx = np.argsort(D, 1)[:, 0:knn+1]
    voted=[]
    for x in range(idx.shape[0]):
        transfered=[reflbl[node_dict[node]] for node in idx[x,].tolist() ]
        vals, counts = np.unique(transfered, return_counts=True)
        order = np.argsort(counts)[::-1]
        voted.append(vals[order[0]])
    inferred=pd.Series(voted,index=querylbl.index)
    LTARI=adjusted_rand_score(querylbl,inferred)
    return LTARI

# get_moran<-function(pos,exp,batch,marker,knn){
#     knnindex=get.knn(pos, k=knn)$nn.index
#     weight=matrix(0,nrow=dim(pos)[1],ncol=dim(pos)[1])
#     for(i in 1:dim(pos)[1]){
#         weight[i,knnindex[i,]]=1
#     }
#     moran=data.frame(moran=0,geary=0,batch=batch)
#     for(marker_id in 1:length(marker)){
#         marker_exp=exp@assays$RNA@data[marker[marker_id],]
#         a=Moran.I(as.vector(marker_exp), weight, scaled = T)$observed
#         b=as.numeric(geary.test(as.vector(marker_exp),mat2listw(weight))$estimate[1])
#         moran=rbind(moran,data.frame(moran=a,geary=b,batch=batch))
#     }
#     moran<-moran[-1,]
#     rownames(moran)<-marker
#     return(moran)
# }
    

import networkx as nx
from scipy.spatial import distance_matrix
import random
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import os
import scipy as sp
import math
#Citation: https://github.com/guott15/SPIRAL/blob/06e4512f70d89e2852b199e5f7542045ff88aa5c/Downstream/metrics_spatial_coherence.ipynb#L313
def create_graph(adata, degree = 4):
        """
        Converts spatial coordinates into graph using networkx library.
        
        param: adata - ST Slice 
        param: degree - number of edges per vertex

        return: 1) G - networkx graph
                2) node_dict - dictionary mapping nodes to spots
        """
        D = distance_matrix(adata.obsm['spatial'], adata.obsm['spatial'])
        # Get column indexes of the degree+1 lowest values per row
        idx = np.argsort(D, 1)[:, 0:degree+1]
        # Remove first column since it results in self loops
        idx = idx[:, 1:]

        G = nx.Graph()
        for r in range(len(idx)):
            for c in idx[r]:
                G.add_edge(r, c)

        node_dict = dict(zip(range(adata.shape[0]), adata.obs.index))
        return G, node_dict
    
#Citation: https://github.com/guott15/SPIRAL/blob/06e4512f70d89e2852b199e5f7542045ff88aa5c/Downstream/metrics_spatial_coherence.ipynb#L313
def generate_graph_from_labels(adata, labels_dict,knn):
    """
    Creates and returns the graph and dictionary {node: cluster_label} for specified layer
    """
    
    g, node_to_spot = create_graph(adata,knn)
    spot_to_cluster = labels_dict

    # remove any nodes that are not mapped to a cluster
    removed_nodes = []
    for node in node_to_spot.keys():
        if (node_to_spot[node] not in spot_to_cluster.keys()):
            removed_nodes.append(node)

    for node in removed_nodes:
        del node_to_spot[node]
        g.remove_node(node)
        
    labels = dict(zip(g.nodes(), [spot_to_cluster[node_to_spot[node]] for node in g.nodes()]))
    return g, labels

#Citation: https://github.com/guott15/SPIRAL/blob/06e4512f70d89e2852b199e5f7542045ff88aa5c/Downstream/metrics_spatial_coherence.ipynb#L313
def spatial_coherence_score(graph, labels):
    g, l = graph, labels
    true_entropy = spatial_entropy(g, l)
    entropies = []
    for i in range(1000):
        new_l = list(l.values())
        random.shuffle(new_l)
        labels = dict(zip(l.keys(), new_l))
        entropies.append(spatial_entropy(g, labels))
        
    return (true_entropy - np.mean(entropies))/np.std(entropies)

#Citation: https://github.com/guott15/SPIRAL/blob/06e4512f70d89e2852b199e5f7542045ff88aa5c/Downstream/metrics_spatial_coherence.ipynb#L313
def spatial_entropy(g, labels):
    """
    Calculates spatial entropy of graph  
    """
    # construct contiguity matrix C which counts pairs of cluster edges
    cluster_names = np.unique(list(labels.values()))
    C = pd.DataFrame(0,index=cluster_names, columns=cluster_names)

    for e in g.edges():
        C[labels[e[0]]][labels[e[1]]] += 1

    # calculate entropy from C
    C_sum = C.values.sum()
    H = 0
    for i in range(len(cluster_names)):
        for j in range(i, len(cluster_names)):
            if (i == j):
                z = C[cluster_names[i]][cluster_names[j]]
            else:
                z = C[cluster_names[i]][cluster_names[j]] + C[cluster_names[j]][cluster_names[i]]
            if z != 0:
                H += -(z/C_sum)*math.log(z/C_sum)
    return H