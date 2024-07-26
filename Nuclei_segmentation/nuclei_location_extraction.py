'''Set of functions to compute the initial nuclei segmentation within the image.
    Edit these functions if you have a preferred segmentation method''' 
import skimage
import numpy as np
from scipy import ndimage as ndi

def compute_DNA_mask(dna_img):
    '''Given the input grayscale DNA image dna_img returns the mask of the segmented objects

        Attributes:

            DNAimg: grayscale image of the DNA channel, numpy array

        Returns:

            label: labelled image of the segmented cells
        '''

    thresh = dna_img > skimage.filters.threshold_triangle(dna_img)*5
    thresh = ndi.binary_erosion(
        thresh, structure=skimage.morphology.disk(3), iterations=2)
    thresh = ndi.binary_fill_holes(thresh)

    label = skimage.measure.label(thresh)

    return label


def retrieve_coordinates(label, cell_size_min=1000, cell_size_max=50000):
    '''Given the labelled image label returns the coordinates of the centroids (center_of_mass) 
       of the nuclei

        Attributes:

            label: image containing labelled segmented objects. Array 
            cell_size_min: Minimum size for the object to be considered a nuclei. int
            cell_size_max: Maximum size for the object to be considered a nuclei. int

        Returns:

            center_of_mass: coordinates of the Center of Mass of the segmented objects, list. 
                Length is the number of objects, width is 2 (X and Y coordinates)
         '''
  
    label_gf = skimage.measure.label(label)
 
    for i in range(1, label_gf.max()+1):
        
        if (np.count_nonzero(label_gf == i) < cell_size_min) or (np.count_nonzero(label_gf == i) > cell_size_max):
            
            label_gf[label_gf == i] = 0
            #print('Nucleus out of range: ',np.count_nonzero(label_gf == i))


    labels = skimage.measure.label(label_gf)
    center_of_mass = np.zeros((label_gf.max(), 2))
    for num in np.arange(np.max(labels)):
        center_of_mass[num] = [ndi.center_of_mass(
            labels == num+1)[0], ndi.center_of_mass(labels == num+1)[1]]
    # remove cells with areas outside of min-max range
    center_of_mass = center_of_mass[~np.all(center_of_mass == 0, axis=1)]
    return center_of_mass
