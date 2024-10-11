'''
Tests for compute_ScaleFEx.py
'''
import cv2 as cv
import numpy as np
import pandas as pd
import os,sys
from warnings import simplefilter
simplefilter(action='ignore',category=pd.errors.PerformanceWarning)
from pathlib import Path
import time
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(ROOT_DIR)

TEST_DIR = os.path.dirname(os.path.realpath(__file__))
# from compute_measurements_functions import *
from compute_ScaleFEx import *

TEST_IMG_DIR = os.path.join(TEST_DIR,'sample_crops')
TEST_MASK_DIR = os.path.join(TEST_DIR ,'primary_masks')
TEST_CSV_DIR = os.path.join(TEST_DIR ,'csv_outputs')
IMG_FILES = sorted([f for f in os.listdir(TEST_IMG_DIR) if f.endswith('.npy')])
MASK_FILES = sorted([f for f in os.listdir(TEST_MASK_DIR) if f.endswith('.npy')])
IMG_ID =3

def test_batch_compute_embeddings():
    print('in') 
    assert Path(os.path.join(TEST_IMG_DIR,IMG_FILES[IMG_ID])).exists()
    print('assertsexists') 
    test_img = np.load(os.path.join(TEST_IMG_DIR,IMG_FILES[IMG_ID]),allow_pickle=True).item()
    test_img_stack = np.stack([test_img['w1'],test_img['w2'],test_img['w3'],
                               test_img['w4'],test_img['w5'],test_img['w6']],axis=0)
    assert test_img['w1'].max() == 24
    assert test_img['w2'].max() == 99
    assert test_img['w3'].max() == 31
    assert test_img['w4'].max() == 49
    assert test_img['w5'].max() == 10
    assert test_img['w6'].max() == 41
    print('asserts img values  done')         
    PARAMS = {'cell_crop':test_img_stack,
              'channel':['w1','w2','w3','w4','w5','w6'],
              'mito_ch':'w5',
              'rna_ch':'w4',
              'downsampling':1,
              'roi':30,
              'visualization':False}
    scalefx = ScaleFEx(**PARAMS)
    sc_vector = scalefx.single_cell_vector
    
    # sc_vector.to_csv(os.path.join(TEST_CSV_DIR,'batch_compute_empeddings_expected_output.csv'),index=False)
    expected_sc_vector = pd.read_csv(os.path.join(TEST_CSV_DIR,'batch_compute_empeddings_expected_output.csv'))
    expected_sc_vector = expected_sc_vector.astype(sc_vector.dtypes.to_dict())
    
    assert sc_vector.iloc[:, 1782].dtype == 'float64',sc_vector.iloc[:, 1782].dtype 
    assert sc_vector.iloc[:, 1781].dtype == 'float64',sc_vector.iloc[:, 1781].dtype 
    assert sc_vector.iloc[:, 1780].dtype == 'int64',sc_vector.iloc[:, 1780].dtype 
    assert sc_vector.iloc[:, 179].dtype == 'float32',sc_vector.iloc[:, 179].dtype 
   
    # print("****************************")
    # print(sc_vector.iloc[:, 0:10].values)
    # print(expected_sc_vector.iloc[:, 0:10].values)
    # print("=============================")
    # print(sc_vector.iloc[:, 1760:1782].values)
    # print(expected_sc_vector.iloc[:, 1760:1782].values)
    # print("-----------------------------")
    # print(len(sc_vector.columns),len(expected_sc_vector.columns))
    # print(sc_vector.iloc[:, 1780])
    # print(expected_sc_vector.iloc[:, 1780])
    # print(sc_vector.iloc[:, 1781])
    # print(expected_sc_vector.iloc[:, 1781])
    # print(sc_vector.iloc[:, 1782])
    # print(expected_sc_vector.iloc[:, 1782])
    # time.sleep(1)
    # pd.testing.assert_frame_equal(expected_sc_vector,sc_vector)
    pd.testing.assert_frame_equal(sc_vector,expected_sc_vector)
