'''Functions that compute each of the ScaleFEx pipeline's features'''

import skimage
import cv2
import numpy as np
from scipy import ndimage as ndi
import pandas as pd
import scipy as sp
import time
import mahotas

def compute_primary_mask(simg):
    '''
    Identify the mask of the input channel image

    Parameters
    ----------
    simg : numpy.ndarray
        Image from which to extract the mask

    Returns
    -------
    labeled_image : numpy.ndarray
        Labeled image (0=background, 1=cell)
    '''
    # Apply a Gaussian filter to reduce noise
    sigma = 1
    ksize = int(6*sigma + 1)  # Ensure ksize is odd
    am = cv2.GaussianBlur(simg, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)

    # Apply a multi-scale filter to extract the cell boundary
    sigma = 1
    ksize = int(6*sigma + 1)  # Ensure ksize is odd
    filter_am = cv2.GaussianBlur(simg, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
    alpha = 10
    am = am + alpha * (am - filter_am)
    am = am > skimage.filters.threshold_multiotsu(am)[0] * 0.9

    # Remove small objects and fill holes
    am = am.astype('uint8')
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    am = cv2.morphologyEx(am, cv2.MORPH_CLOSE, kernel)
    am = ndi.binary_fill_holes(am)

    # Label the image
    labeled_image = skimage.measure.label(am)
    return labeled_image

# ROI param is unneeded
def compute_shape(chan, regions, ROI, segmented_labels):
    """
    Computes shape features from a single region of interest (ROI).

    Args:
        chan (str): channel name
        regions (list): list of regions of interest obtained from skimage.measure.regionprops
        ROI (int): size of the ROI (in pixels)
        segmented_labels (numpy array): segmented image where each pixel corresponds
                                        to a label indicating whether it belongs to the ROI
                                        or not.

    Returns:
        df (pandas dataframe): dataframe containing the computed shape features for the
                               specified ROI. The dataframe has one row and the following columns:
                               'MinRadius_shape' + chan, 'MaxRadius_shape' + chan,
                               'MeanRadius_shape' + chan, 'Area_shape' + chan,
                               'Perimeter_shape' + chan, 'FormFactor_shape' + chan,
                               'Solidity_shape' + chan, 'Extent_shape' + chan,
                               'Eccentricity_shape' + chan, 'Orientation_shape' + chan,
                               'Compactness_shape' + chan
    """

    df = pd.DataFrame([[]], dtype=np.float64)
    df['MinRadius_shape' + chan] = np.min(
        [regions[0].bbox[2]-regions[0].bbox[0], regions[0].bbox[3]-regions[0].bbox[1]])
    df['MaxRadius_shape' + chan] = np.max(
        [regions[0].bbox[2]-regions[0].bbox[0], regions[0].bbox[3]-regions[0].bbox[1]])
    df['MeanRadius_shape' + chan] = regions[0].equivalent_diameter
    df['Area_shape' + chan] = np.nansum(segmented_labels)
    df['Perimeter_shape' + chan] = regions[0].perimeter
    df['FormFactor_shape' + chan] = (
        4 * np.pi * np.nansum(segmented_labels)) / ((regions[0].perimeter) ** 2) + 1e-8
    df['Solidity_shape' + chan] = np.nansum(segmented_labels) / regions[0].convex_area
    df['Extent_shape' + chan] = np.nansum(segmented_labels) / ((2 * ROI) * (2 * ROI))+ 1e-8
    df['Eccentricity_shape' + chan] = regions[0].eccentricity
    df['Orientation_shape' + chan] = regions[0].orientation
    df['Compactness_shape' + chan] = df['MeanRadius_shape' + chan] / np.nansum(segmented_labels)

    return df


def iter_text(chan, simg, segmented_labels, ndistance=5, nangles=4):
    """Computes texture over the specified numbers of pixel distances (ndistance)
    and angles (nangles)
    This function computes the texture of the segmented area of interest (AOI)
    over the specified numbers of pixel distances (ndistance) and angles
    (nangles). The resulting features are ASM (Gray Level Co-occurrence Matrix),
    Contrast, Correlation, Dissimilarity, Homogeneity and Energy for each
    combination of distance and angle. The final result is a data frame with
    the computed features.
    Parameters
    ----------
    chan : str
        Channel name, used as suffix in the resulting data frame.
    simg : 2D array
        Single channel image.
    segmented_labels : 2D array
        Segmented labels of the area of interest.
    ndistance : int, optional
        Number of pixel distances to consider in the computation of the texture
        features. The default is 5.
    nangles : int, optional
        Number of angles to consider in the computation of the texture features.
        The default is 4.
    Returns
    -------
    df : DataFrame
        DataFrame with the computed texture features.
    """
    df = pd.DataFrame([[]])
    angles = np.linspace(0, np.pi, num=nangles)
    distances = np.linspace(5, 5*ndistance, num=ndistance).astype(int)
    for dcount, dis in enumerate(distances):
        for angle in angles:

            texture_props = skimage.feature.graycomatrix(
                np.uint8(simg * segmented_labels) * 255, [dis], [angle])
            df['Texture_dist_' + str(dcount) + 'angle' + str(round(angle, 2)) + chan] = np.nanmean(
                skimage.feature.graycoprops(texture_props, prop='ASM'))
            df['TextContrast_dist_' + str(dcount) + 'angle' + str(round(angle, 2)) + chan] = np.nanmean(
                skimage.feature.graycoprops(texture_props, prop='contrast'))
            df['TextCorrelation_dist_' + str(dcount) + 'angle' + str(round(angle, 2)) + chan] = np.nanmean(
                skimage.feature.graycoprops(texture_props, prop='correlation'))
            df['TextDissimilarity_dist_' + str(dcount) + 'angle' + str(round(angle, 2)) + chan] = np.nanmean(
                skimage.feature.graycoprops(texture_props, prop='dissimilarity'))
            df['TextHomo_dist_' + str(dcount) + 'angle' + str(round(angle, 2)) + chan] = np.nanmean(
                skimage.feature.graycoprops(texture_props, prop='homogeneity'))
            df['TextEnergy_dist_' + str(dcount) + 'angle' + str(round(angle, 2)) + chan] = np.nanmean(
                skimage.feature.graycoprops(texture_props, prop='energy'))
    return df


def texture_single_values(chan, segmented_labels, simg):
    '''Computes global texture values

    Parameters
    ----------
    chan: string
        channel name
    segmented_labels: numpy.ndarray
        segmented labels
    simg: numpy.ndarray
        single channel image

    Returns
    -------
    df: pandas.DataFrame
        dataframe with texture measures
    '''
    df = pd.DataFrame([[]])
    df['Variance_Texture' + chan] = np.nanvar((segmented_labels * simg) > 0) #Variance of the texture of the image (only in the masked area)
    
    df['Variance_Sum_Ave_Texture' + chan] = np.nansum((segmented_labels * simg) > 0) / (
        ((segmented_labels * simg) > 0).shape[0] * ((segmented_labels * simg) > 0).shape[1])+ 1e-8 # Sum of the variance of the texture of the image (only in the masked area)
                                                                                                   # divided by the total number of pixels in the masked area
    df['Variance_Ave_Texture' + chan] = np.nanvar((segmented_labels * simg) > 0) / (
        ((segmented_labels * simg) > 0).shape[0] * ((segmented_labels * simg) > 0).shape[1])+ 1e-8 # Average of the variance of the texture of the image (only in the masked area)
                                                                                                   # divided by the total number of pixels in the masked area
    temp = ((segmented_labels * simg) - 1)/((segmented_labels * simg) - 1).max()
    mask = np.where(temp>=0,1,0).astype(np.uint8)
    temp[temp <0] = 0
    temp = skimage.filters.rank.entropy(skimage.util.img_as_ubyte(temp), skimage.morphology.disk(3),
                                        mask=mask)
    df['Variance_Entropy_Texture' + chan] = np.nanmean(temp > 0) # Entropy of the texture of the image (only in the masked area)
    return df

def granularity(chan, simg, n_convolutions=16):
    '''Computes granularity over a number of n_convolutions'''

    df = pd.DataFrame([[]])

    # Apply grayscale opening to the image
    for gr in range(1, n_convolutions):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (gr, gr))
        df['Granularity_' + chan + '_' +str(gr)] = np.nanmean(cv2.morphologyEx(simg, cv2.MORPH_OPEN, kernel))
    
    return df

def intensity(simg, segmented_labels, chan, regions):
    '''Commputes intensity measurements'''
    
    df = pd.DataFrame([[]])

    nan_image = simg * segmented_labels
    nan_image = nan_image.astype(float)
    nan_image[nan_image == 0] = np.nan

    # The sum of the pixel intensities within an object.
    df['Integrated_intensity_' + chan] = np.nansum(nan_image)

    #: The average pixel intensity within an object.
    df['Mean_intensity_' + chan] = np.nanmean(nan_image)

    #: The standard deviation of the pixel intensities within an object.
    df['Std_intensity_' + chan] = np.nanstd(nan_image)

    #: The maximal pixel intensity within an object.
    df['Max_intensity_' + chan] = np.nanmax(nan_image)

    # The minimal pixel intensity within an object.
    df['Min_intensity_' + chan] = np.nanmin(nan_image)

    # : The intensity value of the pixel for which 75% of the pixels in the object have lower values.
    df['UpperQuartile_intensity_' + chan] = np.nanquantile(nan_image, 0.75)

    #: The intensity value of the pixel for which 25% of the pixels in the object have lower values.
    df['LowerQuartile_intensity_' + chan] = np.nanquantile(nan_image, 0.25)

    #: The median intensity value within the object.
    df['Median_intensity_' + chan] = np.nanmedian(nan_image)

    edge_image = (np.sqrt(cv2.Sobel(segmented_labels.astype(np.uint8), 
                    cv2.CV_64F, 1, 0, ksize=3)**2 + cv2.Sobel(segmented_labels.astype(np.uint8), 
                    cv2.CV_64F, 0, 1, ksize=3)**2)!= 0) * simg
    
    edge_image = edge_image.astype(float)
    edge_image[edge_image == 0] = np.nan

    #: The sum of the edge pixel intensities of an object.
    df['Integrated_intensityEdge_' + chan] = np.nansum(edge_image)

    #: The average edge pixel intensity of an object.
    df['Mean_intensityEdge_' + chan] = np.nanmean(edge_image)

    df['Std_intensityEdge_' + chan] = np.nanstd(
        edge_image)  #: The standard deviation of the edge pixel intensities of an object.

    #: The maximal edge pixel intensity of an object.
    df['Max_intensityEdge_' + chan] = np.nanmax(edge_image)

    #: The minimal edge pixel intensity of an object.
    df['Min_intensityEdge_' + chan] = np.nanmin(edge_image)

    #: The distance between the centers of gravity in the gray-level representation of the object 
    # and the binary representation of the object.
    df['MassDisplacement_intensity' + chan] = sp.spatial.distance.pdist(
        [ndi.center_of_mass(simg, segmented_labels), regions[0].centroid])[0]

    #: The median absolute deviation (MAD) value of the intensities within the object.
    # The MAD is defined as the median(|xi - median(x)|).
    df['MAD_intensity_' +
        chan] = np.nanmedian(np.abs(nan_image-df['Median_intensity_' + chan].values))

    # , Location_CenterMassIntensity_Y: The (X,Y) coordinates of the intensity weighted centroid (= center of mass = first moment) 
    # of all pixels within the object.
    df['Location_CenterMass_intensity_X' +
        chan] = ndi.center_of_mass(simg, segmented_labels)[0]

    #: The (X,Y) coordinates of the pixel with the maximum intensity within the object.
    df['Location_CenterMass_intensity_Y' +
        chan] = ndi.center_of_mass(simg, segmented_labels)[1]

    return df


def create_concentric_areas(scale, fact, ROI, DAPI=0):
    '''Creates the concentric areas for subsequent measurements'''

    P = np.zeros((int(ROI)*2, int(ROI)*2))
    P2 = np.zeros((int(ROI)*2, int(ROI)*2))
    if DAPI == 0:
        Dma = fact*(ROI/scale)
    else:
        Dma = fact*(DAPI)
    cv2.circle(P, (int(np.ceil(ROI)), int(np.ceil(ROI))), int(Dma), 1, -1)
    if fact != 0:
        cv2.circle(P2, (int(np.ceil(ROI)), int(np.ceil(ROI))),
                   int(Dma + Dma), 1, -1)
    else:
        cv2.circle(P2, (int(np.ceil(ROI)), int(np.ceil(ROI))),
                   int((ROI/scale)), 1, -1)

    return(P2 - P)


def concentric_measurements(scale, ROI, simg, segmented_labels, chan, DAPI=0):


    imgConc = {}
    Pt = {}
    df = pd.DataFrame([[]])
    for fact in range(0, scale):
        if DAPI != 0:
            P = create_concentric_areas(scale, fact, ROI, DAPI=DAPI)
        else:
            P = create_concentric_areas(scale, fact, ROI)

        imgConc[fact] = (P * (simg * segmented_labels))
        Pt[fact] = np.nansum(P * segmented_labels)

        if np.nansum(Pt[fact]) > 0:
            df['Concent_tot_intensity_' +
                str(fact) + chan] = np.nansum(imgConc[fact]) / np.nansum(P)+ 1e-8
            df['Concent_mean_intensity_' +
                str(fact) + chan] = np.nanmean(imgConc[fact]) / np.nansum(P)+ 1e-8
            df['Concent_variation_intensity_' +
                str(fact) + chan] = np.nanstd(imgConc[fact]) / np.nansum(P)+ 1e-8
            for gr in range(1, 16):
                df['Granularity_' + chan + '_' + str(gr) + '_Conc' + str(fact)] = np.nanmean(
                    cv2.morphologyEx((imgConc[fact] / Pt[fact]).astype(np.float32), cv2.MORPH_OPEN, 
                                     cv2.getStructuringElement(cv2.MORPH_RECT, (gr, gr))))
        else:
            df['Concent_tot_intensity_' + str(fact) + chan] = 0
            df['Concent_mean_intensity_' + str(fact) + chan] = 0
            df['Concent_variation_intensity_' + str(fact) + chan] = 0

            for gr in range(1, 16):
                df['Granularity_' + chan + '_' +
                    str(gr) + '_Conc' + str(fact)] = 0

    for fact in range(1, 7):

        for fact2 in range(fact + 1, 8):

            if np.nansum(Pt[fact]) > 0 and np.nansum(Pt[fact2]) > 0:
                RR = np.concatenate([np.asarray(imgConc[fact]).reshape(-1, ) / Pt[fact],
                                    np.asarray(imgConc[fact2]).reshape(-1, ) / Pt[fact2]])
                df['Concent_Radial_intensity_' + str(fact) + '_' + str(fact2) + '_' + chan] = np.nanmean(
                    sp.stats.variation(RR))
            else:
                df['Concent_Radial_intensity_' +
                    str(fact) + '_' + str(fact2) + '_' + chan] = 0

    return df

def zernike_measurements(segmented_labels, roi, chan):
    '''Computes the features and appends them into a single line vector'''
    
    df = pd.DataFrame([[]])

    zernike_moments = mahotas.features.zernike_moments(segmented_labels, roi)
    for m,z in enumerate(zernike_moments):
        df['Zernike_' + chan + '_' + str(m)] = z

    return df

def show_cells(images, title=[''], size=3):
    ''' Function to visualize  images in a compact way '''
    import matplotlib.pyplot as plt
    _, ax = plt.subplots(1, len(images), figsize=(int(size*len(images)), size))
    if len(images) > 1:
        for i,_ in enumerate(images):
            ax[i].imshow(images[i], cmap='Greys_r')
            if len(title) == len(images):
                ax[i].set_title(title[i])
            else:
                ax[0].set_title(title[0])
            ax[i].axis('off')
    else:
        ax.imshow(images[0])
        ax.set_title(title[0])
        ax.axis('off')
    plt.show()

def mitochondria_measurement(segmented_labels, simg, viz=False):


    df = pd.DataFrame([[]])

    sigma = 1
    ksize = int(6*sigma + 1)  # Ensure ksize is odd
    mito = cv2.GaussianBlur(simg, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)

    sigma = 3
    ksize = int(6*sigma + 1)  # Ensure ksize is odd
    filter_mito = cv2.GaussianBlur(mito, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
    alpha = 40
    mito = mito + alpha * (mito - filter_mito)

    mito_segmented = mito > skimage.filters.threshold_multiotsu(mito)[-1]
    mito_segmented = mito_segmented*segmented_labels
    skel = skimage.morphology.skeletonize(mito_segmented)
    labeled_skeleton = skimage.morphology.label(skel)
    for u in range(1, np.max(labeled_skeleton)):
        if np.nansum(labeled_skeleton == u) < 5:
            labeled_skeleton[labeled_skeleton == u] = 0
    labeled_skeleton = skimage.morphology.label(labeled_skeleton)
    if np.count_nonzero(labeled_skeleton) < 1:
        df['MitoCount'] = 0
        df['MitoVolumeMean'] = 0
        df['MitoVolumeTot'] = 0
        df['MitoVolumeSkel'] = 0
        branch = []
        aspect_ratio = []
        end_points = []
    else:
        df['MitoCount'] = np.max(labeled_skeleton)
        df['MitoVolumeMean'] = np.count_nonzero(mito_segmented) / np.max(labeled_skeleton)+ 1e-8
        df['MitoVolumeTot'] = np.count_nonzero(mito_segmented)
        df['MitoVolumeSkel'] = np.count_nonzero(labeled_skeleton)

        k_diag_upslope = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
        K_diag_downslope = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        structure1 = np.array([[0, 1, 0],
                               [1, 1, 1],
                               [0, 1, 0]], dtype=np.uint8)
        branch = []
        aspect_ratio = []
        end_points = []

        for ii in range(1, np.max(labeled_skeleton)):
            SS = labeled_skeleton == ii
            SS = SS.astype(np.uint8)
            reg = skimage.measure.regionprops(SS * 1)
            if reg[0].minor_axis_length > 0:
                aspect_ratio.append(
                    reg[0].major_axis_length / reg[0].minor_axis_length+ 1e-8)
            else:
                aspect_ratio.append(reg[0].major_axis_length / 1)
            a_orthog = SS.copy()
            B = cv2.filter2D(SS, -1, k_diag_upslope, borderType=cv2.BORDER_CONSTANT)
            a_orthog[B == 2] = 1
            B = cv2.filter2D(SS, -1, K_diag_downslope, borderType=cv2.BORDER_CONSTANT)
            a_orthog[B == 2] = 1
            B = cv2.filter2D(a_orthog, -1, structure1, borderType=cv2.BORDER_CONSTANT)
            image_of_branch_points = B >= 4
            branch.append(skimage.morphology.label(
                image_of_branch_points).max())
            B = np.zeros_like(SS)
            for k in range(9):
                K = np.zeros((9))
                K[4] = 1
                K[k] = 1
                B = B + cv2.filter2D(SS.astype(np.float32), -1, K.reshape(3, 3).astype(np.float32), borderType=cv2.BORDER_CONSTANT)
                
            end_points.append(np.count_nonzero(B == 10))

    if branch:
        df['MitoMeanBranch'] = np.nanmean(branch)
        df['MitoStdBranchN'] = np.nanstd(branch)
        df['MitoUquanBranchN'] = np.nanquantile(branch, 0.75)
        df['MitoLQuanBranchN'] = np.nanquantile(branch, 0.25)
        df['MitoMedianBranchN'] = np.nanmedian(branch)
    else:
        df['MitoMeanBranch'] = 0
        df['MitoStdBranchN'] = 0
        df['MitoUquanBranchN'] = 0
        df['MitoLQuanBranchN'] = 0
        df['MitoMedianBranchN'] = 0

    if aspect_ratio or end_points:
        df['MitoMeanAspectRatio'] = np.nanmean(aspect_ratio)
        df['MitoStdAspectRatio'] = np.nanstd(aspect_ratio)
        df['MitoUquanAspectRatio'] = np.nanquantile(aspect_ratio, 0.75)
        df['MitoLQuanAspectRatio'] = np.nanquantile(aspect_ratio, 0.25)
        df['MitoMedianAspectRatio'] = np.nanmedian(aspect_ratio)
        df['MitoMeanEndPoints'] = np.nanmean(end_points)
        df['MitoStdEndPointsN'] = np.nanstd(end_points)
        df['MitoUquanEndPointsN'] = np.nanquantile(end_points, 0.75)
        df['MitoLQuanEndPointsN'] = np.nanquantile(end_points, 0.25)
        df['MitoMedianEndPointsN'] = np.nanmedian(end_points)
        if viz is True:
            show_cells([mito_segmented, skel], title=['Mito', 'skeleton'])
    else:

        df['MitoMeanAspectRatio'] = 0
        df['MitoStdAspectRatio'] = 0
        df['MitoUquanAspectRatio'] = 0
        df['MitoLQuanAspectRatio'] = 0
        df['MitoMedianAspectRatio'] = 0
        df['MitoMeanEndPoints'] = 0
        df['MitoStdEndPointsN'] = 0
        df['MitoUquanEndPointsN'] = 0
        df['MitoLQuanEndPointsN'] = 0
        df['MitoMedianEndPointsN'] = 0


    return df


def RNA_measurement(segmented_labels, simg, viz=False):
    '''Measure RNA features'''
   
    df = pd.DataFrame([[]])
    sigma = 1
    ksize = int(6*sigma + 1)  # Ensure ksize is odd
    Rn = cv2.GaussianBlur(simg, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
    
    sigma = 1
    ksize = int(6*sigma + 1)
    filter_Rn = cv2.GaussianBlur(Rn,(ksize, ksize), sigmaX=sigma, sigmaY=sigma)
    alpha = 30
    Rn = Rn + alpha * (Rn - filter_Rn)
    Rn = Rn*cv2.erode((segmented_labels * 1).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10)), iterations=1)
    Rn = Rn > skimage.filters.threshold_otsu(Rn[Rn > 1]) * 1.1
    Rn = ndi.binary_opening(Rn, skimage.morphology.disk(1))
    Rn = ndi.binary_closing(Rn)
    for u in range(1, np.max(Rn)):
        if np.nansum(Rn == u) < 5:
            Rn[Rn == u] = 0

    Rn = skimage.measure.label(Rn)
    if np.count_nonzero(Rn) < 1:

        df['RNACount'] = 0
        df['RNAVolumeMean'] = 0
        df['RNameanDistance'] = 0
        df['RNAStdDistance'] = 0
        df['RNAUquantDistance'] = 0
        df['RNALquanrDistance'] = 0
        df['RNamedianDistance'] = 0
        return df
    if viz is True:
        show_cells([Rn], title=['RNA'])

    if Rn.max() > 0:
        df['RNACount'] = np.max(Rn)
        df['RNAVolumeMean'] = np.count_nonzero(Rn) / np.max(Rn)+ 1e-8
        dist_among_rna = []
        for tt in range(1, np.max(Rn) + 1):
            for t in range(tt, np.max(Rn) + 1):
                dist_among_rna.append(sp.spatial.distance.pdist([ndi.center_of_mass(1 * (Rn == tt)),
                                                          ndi.center_of_mass(1 * (Rn == t))]))
        df['RNameanDistance'] = np.nanmean(dist_among_rna)
        df['RNAStdDistance'] = np.nanstd(dist_among_rna)
        df['RNAUquantDistance'] = np.nanquantile(dist_among_rna, 0.75)
        df['RNALquanrDistance'] = np.nanquantile(dist_among_rna, 0.25)
        df['RNamedianDistance'] = np.nanmedian(dist_among_rna)
    else:
        df['RNACount'] = 0
        df['RNAVolumeMean'] = 0
        df['RNameanDistance'] = 0
        df['RNAStdDistance'] = 0
        df['RNAUquantDistance'] = 0
        df['RNALquanrDistance'] = 0
        df['RNamedianDistance'] = 0


    return df


def correlation_measurements(simgi, simgj, chan, chanj, Labi, Labj):
    '''Measure correlation between channels'''
   
    df = pd.DataFrame([[]])
    correlation_coefficient = np.nanmean(np.corrcoef(simgi, simgj))
    df['Correlation_' + chan + '_' + chanj] = correlation_coefficient
    # slope value= slope.slope
    slope = sp.stats.linregress(simgi.reshape(-1, ), simgj.reshape(-1, ))
    df['Correlation_Slope_' + chan + '_' + chanj] = slope.slope
    overlap_coeff = np.nansum(
        simgi * simgj) / np.sqrt(np.nansum(simgi * simgi) * np.nansum(simgj * simgj))
    df['Correlation_Overlap_' + chan + '_' + chanj] = overlap_coeff
    M1 = np.nanmean(sum(simgi * Labj) / (sum(simgi)+ 1e-8))
    M2 = np.nanmean(sum(simgj * Labi) / (sum(simgj)+ 1e-8))
    df['Correlation_Mander1_' + chan + '_' + chanj] = M1
    df['Correlation_Mander2_' + chan + '_' + chanj] = M2
    Rmax = np.max([len(np.unique(simgi)), len(np.unique(simgj))])  # Aux
    Di = abs(len(np.unique(simgi)) - len(np.unique(simgj)))  # Aux
    Wi = (Rmax - Di) / Rmax+ 1e-8  # Aux
    RWC1 = sum(sum(simgi * Labj * Wi) / (sum(simgi)+ 1e-8))
    RWC2 = sum(sum(simgj * Labi * Wi) / (sum(simgj)+ 1e-8))
    df['Correlation_RWC1_' + chan + '_' + chanj] = RWC1
    df['Correlation_RWC2_' + chan + '_' + chanj] = RWC2

    return df

def single_cell_feature_extraction(simg, channels, roi, mito_ch, rna_ch, downsampling, viz):
    '''Computes the features and appends them into a single line vector'''
    simg = simg.squeeze().transpose(1, 2, 0)
    segmented_labels = {}
    regions = {}
    measurements = pd.DataFrame([{}])

    for i, chan in enumerate(channels):
        segmented_labels[i] = compute_primary_mask(simg[:, :, i])

        if i == 0:
            orig_nuclei = segmented_labels[i]
            nn = segmented_labels[i][roi, roi]
            segmented_labels[i] = segmented_labels[i] == nn
        else:
            nn = segmented_labels[i][int(roi/2):int(roi*(3/2)), int(roi/2):int(roi*(3/2))]
            try:
                nn = np.bincount(nn[nn > 0]).argmax()
            except ValueError:
                print('out except for size inconsistency')
                return False, False
            segmented_labels[i] = segmented_labels[i] == nn

        invMask = segmented_labels[i] < 1

        if viz:
            show_cells([simg[:, :, i], segmented_labels[i]], title=[chan + '_'+str(i), 'mask'])

        if np.count_nonzero(segmented_labels[i]) <= 50/downsampling:
            print('out size')
            return False, False

        a = simg[:, :, i]*segmented_labels[i]
        b = a[a > 0]
        a = simg[:, :, i]*invMask
        c = a[a > 0]

        SNR = np.mean(b)/(np.std(c)+1e-8)
        measurements['SNR_intensity' + chan] = SNR
        regions[i] = skimage.measure.regionprops(segmented_labels[i].astype(int))

        # Shape
        shape_df = compute_shape(chan, regions[i], roi, segmented_labels[i])
        measurements = pd.concat([measurements, shape_df], axis=1)

        # Texture
        texture_df = iter_text(chan, simg[:, :, i], segmented_labels[i], ndistance=5, nangles=4)
        measurements = pd.concat([measurements, texture_df], axis=1)

        # Granularity
        granularity_df = granularity(chan, simg[:, :, i], n_convolutions=16)
        measurements = pd.concat([measurements, granularity_df], axis=1)

        # Intensity
        intensity_df = intensity(simg[:, :, i], segmented_labels[i], chan, regions[i])
        measurements = pd.concat([measurements, intensity_df], axis=1)

        # Concentric measurements
        scale = 8
        if chan == channels[0]:
            nuc = measurements['MaxRadius_shape'+channels[0]].values[0] * 0.1
        else:
            nuc = 0
        concentric_df = concentric_measurements(scale, roi, simg[:, :, i], segmented_labels[i], chan, DAPI=nuc)
        measurements = pd.concat([measurements, concentric_df], axis=1)

        # Zernike measurements
        zernike_df = zernike_measurements(segmented_labels[i], roi, chan)
        measurements = pd.concat([measurements, zernike_df], axis=1)

        # Mitochondria measurements
        if chan == mito_ch:
            mitochondria_df = mitochondria_measurement(segmented_labels[i], simg[:, :, i], viz=viz)
            measurements = pd.concat([measurements, mitochondria_df], axis=1)

        # RNA measurements
        if chan == rna_ch:
            rna_df = RNA_measurement(segmented_labels[0], simg[:, :, i], viz=viz)
            measurements = pd.concat([measurements, rna_df], axis=1)

    # Colocalization
    for i, chan in enumerate(channels):
        for j in range(i + 1, len(channels)):
            colocalization_df = correlation_measurements(
                simg[:, :, i], simg[:, :, j], chan, channels[j], segmented_labels[i], segmented_labels[j]
            )
            measurements = pd.concat([measurements, colocalization_df], axis=1)

    quality_flag = True
    return quality_flag, measurements

# Define additional functions with print statements similarly, if needed.


