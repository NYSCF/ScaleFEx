'''Unit tests for compute_global_values.py'''
import skimage
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import pytest

import sys, os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)


from compute_global_values import *

TEST_IMG_DIR = os.path.join(ROOT_DIR,'tests','test_imgs')
# @pytest.mark.skip(reason="not yet implemented")
def test_calculateQC():
    img = np.load(os.path.join(TEST_IMG_DIR,'rxrx2_O24_s2.npy'),allow_pickle=True).item()
    channels = list(img.keys())
    # print(img)
    img_raw = np.zeros((len(channels),img['w1'].shape[0],img['w1'].shape[1]))
    for i,ch in enumerate(channels):
        img_raw[i,:,:] = img[ch]
    PARAMS = {"tot_n": 20,
              "experiment_name":"rxrx2",
              "img_raw": img_raw,
              "well":"O24",
              "plate":"Plate1",
              "site":"s2",
              "channel":list(img.keys()),
              "indQC":0}
    
    df, indQC = calculateQC(**PARAMS)
    assert not df.isna().any().any()
    assert not df.isin([np.inf,-np.inf]).any().any()

# @pytest.mark.skip(reason="not yet implemented")
def test_calculateQC_OOF():
    img = np.load(os.path.join(TEST_IMG_DIR,'rxrx2_O24_s2_OOF.npy'),allow_pickle=True).item()
    channels = list(img.keys())
    img_raw = np.zeros((len(channels),img['w1'].shape[0],img['w1'].shape[1]))
    for i,ch in enumerate(channels):
        img_raw[i,:,:] = img[ch].astype(float)
    print(np.unique(img_raw[0]))
    PARAMS = {"tot_n": 20,
              "experiment_name":"rxrx2",
              "img_raw": img_raw,
              "well":"O24",
              "plate":"Plate1",
              "site":"s2",
              "channel":list(img.keys()),
              "indQC":0}
   
    df, indQC = calculateQC(**PARAMS)
    assert not df.isna().any().any()
    assert not df.isin([np.inf,-np.inf]).any().any()
