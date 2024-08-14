'''Main class for the ScaleFEx feature computations'''

import warnings
import pandas as pd
import sys
ROOT_DIR = '/'.join(__file__.split('/')[:-1])
sys.path.append(ROOT_DIR)
from compute_measurements_functions import single_cell_feature_extraction

warnings.filterwarnings('ignore')


class ScaleFEx:

    """ Pipeline to extract a vector of fixed features from a HCI screen

        Attributes: 

        exp_folder = string containing the experiment folder location (eg folder with subfolders of plates), str
        experiment_name = experiment name (eg ScaleFEx_xx), str
        saving_folder = string containing the destination folder, str 
        plates = plates IDs to be analysed, as they appear in the pathname, list of strings 
        channel = channel IDs to be analysed, as they appear in the pathname. 
            NOTE: the nuclei stain has to be the firs channel of the list. list of strings 
        img_size = x and y size of the image, list of ints
        roi = half the size of the cropped area around the cell, int
        parallel = use multiprocessing to analyse each plate with a worker, Bool
        save_image = Specify if to save the cropped images as a .npy file. False if not,
            pathname of saving location if yes
        stack = performs max projection on images if the acquisition mode was multi-stack, Bool
        CellType = choose between 'Fib' (Fibroblasts), 'IPSC' or 'Neuron'. 
            A different segmentation algorithm is used based on this choice. str
        mito_ch = which of the channels is mito_chondria (if any), str
        rna_ch = which of the channels is RNA (if any), str


    """

    def __init__(self, cell_crop, channel=['ch4', 'ch1', 'ch2', 'ch3', 'ch5'],
                 mito_ch='ch2', rna_ch='ch5', downsampling=1, visualization=False, roi=150):
        
        self.channel = channel
        self.mito_ch = mito_ch
        self.rna_ch = rna_ch
        self.downsampling = downsampling
        self.viz = visualization
        self.roi = int(roi)

        self.load_preprocess_and_compute_feature(cell_crop)



    def load_preprocess_and_compute_feature(self, cell_crop):
        ''' Function that imports the images and extracts the location of cells'''
        qc_flag, sc_vec = single_cell_feature_extraction(cell_crop, self.channel,self.roi,
                                    self.mito_ch,self.rna_ch,self.downsampling,self.viz)
        self.single_cell_vector = pd.DataFrame(sc_vec)
