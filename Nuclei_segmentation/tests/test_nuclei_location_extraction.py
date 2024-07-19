import skimage
import numpy as np
from scipy import ndimage as ndi
import pytest
import cv2 as cv
import os,sys
ROOT_DIR = '/'.join(__file__.split('/')[:-2])
TEST_DIR = os.path.join(ROOT_DIR,'tests')
sys.path.append(ROOT_DIR)
from nuclei_location_extraction import *

# @pytest.mark.skip(reason='not implemented yet')
def test_compute_DNA_mask():
    dna_img = cv.imread(os.path.join(TEST_DIR,'sample_data/rxrx2_Plate1_G02_s2_w1.png'),cv.IMREAD_UNCHANGED)
    
    label = compute_DNA_mask(dna_img)
    # cv.imwrite('tests/sample_data/rxrx2_Plate1_G02_s2_w1_DNA_mask.tiff',label)
    expected_label = cv.imread(os.path.join(TEST_DIR,'sample_data/rxrx2_Plate1_G02_s2_w1_DNA_mask.tiff'),cv.IMREAD_UNCHANGED)
    assert len(np.unique(label)) == len(np.unique(expected_label))
    assert not np.isnan(np.min(label))
    assert label is not None
    assert label.shape == dna_img.shape

# @pytest.mark.skip(reason='not implemented yet')
def test_retrieve_coordinates():
    label = cv.imread(os.path.join(TEST_DIR,'sample_data/rxrx2_Plate1_G02_s2_w1_DNA_mask.tiff'),cv.IMREAD_UNCHANGED)
    coords = retrieve_coordinates(label,cell_size_min=50)

    expected_coords = np.load(os.path.join(TEST_DIR,'sample_data/expected_coords.npy'),allow_pickle=True)
    np.testing.assert_array_equal(coords,expected_coords)
    assert not np.isnan(np.min(coords))
    assert coords is not None
    assert coords.shape[1] == 2
     

