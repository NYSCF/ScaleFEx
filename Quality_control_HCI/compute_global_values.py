'''Functions for computing site-level image quality control metrics '''
import skimage
import cv2
import pandas as pd
import numpy as np
from scipy import ndimage as ndi

def calculateQC(tot_n,live_cells,experiment_name,img_raw,well,plate,site,channel,indQC,neural_tracing):
    '''
    Computes site-level image quality control metrics

    Attributes:
        tot_n: int
            total number of cells

        live_cells: int
            total number of live cells

        experiment_name: str
            name of the experiment

        img_raw: np.ndarray
            raw image

        well: str
            well of the image
        
        site: str
            site of well of the image
        
        plate: str
            plate of the image

        channel: list
            list of channels

        indQC: int
            index of row in QC dataframe to write computed metrics

        neural_tracing: str
            name of channel to trace neurites (if any); if None, left empty

    Returns:
        Parameters_final: pd.DataFrame
            dataframe with computed metrics
    '''
    Parameters_final=pd.DataFrame()
    for ch,img in enumerate(img_raw):
        Parameters=pd.DataFrame()
        Parameters.loc[indQC,'Well']=well
        Parameters.loc[indQC,'channel']=channel[ch]
        Parameters.loc[indQC,'Site']=site
        Parameters.loc[indQC,'Plate']=plate
        Parameters.loc[indQC,'batch']=experiment_name
        #ax[n].imshow(img)
        #ax[n].set_title(chan)
        Parameters.loc[indQC,'Empty']=False
        Parameters.loc[indQC,'Usable']=True
        if live_cells==0:
            Parameters.loc[indQC,'Empty']=True
            Parameters.loc[indQC,'Usable']=False
            #plt.imshow(img)
        
        Parameters.loc[indQC,'tot_cell_num']=tot_n
        Parameters.loc[indQC,'Cell_num']=live_cells
        Parameters.loc[indQC,'Max_Intensity']=np.max(img)
        Parameters.loc[indQC,'Min_Intensity']=np.min(img)
        Parameters.loc[indQC,'Mean_Intensity']=np.mean(img)
        
        # for blur detection:
        blur_img = img.copy()
        blur_img[np.percentile(blur_img,99.9)>blur_img] = np.percentile(blur_img,99.9)
        blur_img = cv2.resize(blur_img,(1080,1080),cv2.INTER_AREA)

        Parameters.loc[indQC,'Blur']=(1-cv2.Laplacian((blur_img-blur_img.min())/(img-img.min()).max(), cv2.CV_64F,ksize=9)\
                                      *np.exp(-1/(tot_n+np.finfo(float).eps))).var()
        # threshold chosen based on nuclei channel experiment in BlurTest.ipynb
        if Parameters.loc[indQC,'Blur']>2700:
            Parameters.loc[indQC,'InFocus']=True
        else:
            Parameters.loc[indQC,'InFocus']=False
            Parameters.loc[indQC,'Usable']=False

        if (np.isnan(img).all()==False) and (len(np.unique(img)))>1:
            Thres=skimage.filters.threshold_otsu(img)#*0.9
            a=img*(img>Thres)
            #ax[n+5].imshow(a)
            b=a[a>0]
            Parameters.loc[indQC,'Mean_Foreground_Intensity']=np.mean(b)
            
            b=b-np.min(img)
            b=b/np.max(img)
            high=np.mean(b)
            a=img*(img<Thres)
            b=a[a>0]
            Parameters.loc[indQC,'Mean_Background_Intensity']=np.mean(b)
            
            b=b-np.min(img)
            b=b/np.max(img)
            low=np.std(b)
            Parameters.loc[indQC,'SNR']=high/(low+np.finfo(float).eps)
        else:
            Parameters.loc[indQC,'Mean_Foreground_Intensity']=np.nan
        
            Parameters.loc[indQC,'Mean_Background_Intensity']=np.nan

            Parameters.loc[indQC,'SNR']=np.nan
        if ch==0:
            Parameters.loc[indQC,'neural_len']=0
        if neural_tracing==channel[ch]:
            Parameters.loc[indQC,'neural_len']=compute_axons(img).sum()
        else:
            
            Parameters.loc[indQC,'neural_len']=0


        indQC+=1
        # #clear_output(wait=True)
        # Parameters.to_csv(csv_fileQC[:-4]+'.csv',mode='a',header=flag)
        # flag=False
        Parameters_final=pd.concat([Parameters_final,Parameters],axis=0)


    return Parameters_final,indQC

def compute_axons(img):
    '''Compute the skeleton of an image with neurites'''
    if np.max(img)<1.1:
        img=img*255
    img=img.astype('uint8')
    trace = ndi.gaussian_filter((img), 1)
    somamask=img > skimage.filters.threshold_multiotsu(img)[-1]
    somamask=ndi.binary_opening(somamask,iterations=3)
    somamask=1-somamask
    filter_trace = ndi.gaussian_filter(trace, 1)
    alpha = 50
    trace = trace + alpha * (trace - filter_trace)

    trace_segmented = trace > skimage.filters.threshold_multiotsu(img)[0]
    trace_segmented=ndi.binary_opening(trace_segmented,structure=skimage.morphology.disk(1),iterations=1)
    # trace_segmented=ndi.binary_closing(trace_segmented,structure=skimage.morphology.disk(1),iterations=1)
    
    #skel2=ndi.binary_closing(skel2,iterations=2)
    #skel2=ndi.binary_dilation(skel2,iterations=2)
    #skel2 = skimage.morphology.skeletonize(skel2)
    trace_segmented=skimage.feature.canny(ndi.gaussian_filter((img), 1))
    trace_segmented=ndi.binary_closing(trace_segmented,structure=skimage.morphology.disk(1),iterations=2)
    #trace_segmented=ndi.binary_fill_holes(trace_segmented)
    skel = skimage.morphology.thin(trace_segmented)
    skel=skel*somamask
    return skel 
