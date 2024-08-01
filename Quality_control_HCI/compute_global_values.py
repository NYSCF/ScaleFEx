'''Functions for computing site-level image quality control metrics '''
import skimage
import cv2
import pandas as pd
import numpy as np
from scipy import ndimage as ndi

def calculateQC(tot_n,experiment_name,img_raw,well,plate,site,channel,indQC):
    '''
    Attributes:
        tot_n: int

        experiment_name: str

        img_raw: np.ndarray

        well: str

        plate: str

        site: str

        channel: list

        indQC: int

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
        Parameters.loc[indQC,'tot_cell_num']=tot_n
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

        indQC+=1
        # #clear_output(wait=True)
        # Parameters.to_csv(csv_fileQC[:-4]+'.csv',mode='a',header=flag)
        # flag=False
        Parameters_final=pd.concat([Parameters_final,Parameters],axis=0)


    return Parameters_final,indQC


def retrieve_coordinates(label):
    '''Given the labelled image label returns the coordinates of the centroids (center_of_mass) 
       of the nuclei

        Attributes:

        label: image containing labelled segmented objects. Array 
        
        Returns:

        center_of_mass: coordinates of the Center of Mass of the segmented objects, list. 
             Length is the number of objets, width is 2 (X and Y coordinates)
         '''
    center_of_mass = []
    if np.max(label)==1:
        label=skimage.measure.label(label)
    if label.size>0:
        
        for num in np.arange(1,np.max(label)+1):
            if num in label:
                
                coordinates = ndi.measurements.center_of_mass(label == num) 
                center_of_mass.append([coordinates[0], coordinates[1]])

    return center_of_mass