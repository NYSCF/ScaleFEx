import skimage
import numpy as np
import pandas as pd
import pytest 
import os,sys
import warnings
warnings.simplefilter(action='ignore',category=pd.errors.PerformanceWarning)

ROOT_DIR = '/'.join(__file__.split('/')[:-2])
sys.path.append(ROOT_DIR)
from compute_measurements_functions import *
TEST_DIR = "/".join(__file__.split('/')[:-1])
TEST_IMG_DIR = os.path.join(TEST_DIR,'sample_crops')
TEST_MASK_DIR = os.path.join(TEST_DIR ,'primary_masks')
TEST_CSV_DIR = os.path.join(TEST_DIR ,'csv_outputs')
IMG_FILES = sorted([f for f in os.listdir(TEST_IMG_DIR) if f.endswith('.npy')])
MASK_FILES = sorted([f for f in os.listdir(TEST_MASK_DIR) if f.endswith('.npy')])
IMG_ID =3

# @pytest.mark.skip(reason="not implemented yet")
def test_compute_primary_mask():
    test_img = np.load(os.path.join(TEST_IMG_DIR,IMG_FILES[IMG_ID]),allow_pickle=True).item()
    test_mask = np.load(os.path.join(TEST_MASK_DIR,MASK_FILES[IMG_ID]),allow_pickle=True).item()
    primary_mask = compute_primary_mask(test_img['w1'])
    assert primary_mask.shape == test_img['w1'].shape
    assert primary_mask.min() == 0
    assert np.array_equal(primary_mask,test_mask['w1'])

@pytest.mark.skip(reason="under development")
def test_compute_primary_mask_neuron():
    pass

# @pytest.mark.skip(reason="not implemented yet")
def test_compute_shape():
    channel = 'w2'
    # test_img = np.load(os.path.join(TEST_IMG_DIR,IMG_FILES[IMG_ID]),allow_pickle=True).item()
    test_mask = np.load(os.path.join(TEST_MASK_DIR,MASK_FILES[IMG_ID]),allow_pickle=True).item()
    regions = skimage.measure.regionprops(test_mask[channel].astype(int))
    
    df = compute_shape(channel,regions,ROI=30,segmented_labels=test_mask[channel])
    expected_df = pd.read_csv(os.path.join(TEST_CSV_DIR,'compute_shape_output.csv'))
    pd.testing.assert_frame_equal(df,expected_df)

# @pytest.mark.skip(reason="no features extracted from crop?")
def test_iter_text():
    channel = 'w3'
    test_img = np.load(os.path.join(TEST_IMG_DIR,IMG_FILES[IMG_ID]),allow_pickle=True).item()
    test_mask = np.load(os.path.join(TEST_MASK_DIR,MASK_FILES[IMG_ID]),allow_pickle=True).item()

    df = iter_text(channel,test_img[channel],test_mask[channel],ndistance=5,nangles=4)
    # print(df)
    expected_df = pd.read_csv(os.path.join(TEST_CSV_DIR,'iter_text_output.csv'))
    pd.testing.assert_frame_equal(df,expected_df)

# @pytest.mark.skip(reason="not implemented yet")
def test_texture_single_values():
    channel = 'w3'
    test_img = np.load(os.path.join(TEST_IMG_DIR,IMG_FILES[IMG_ID]),allow_pickle=True).item()
    test_mask = np.load(os.path.join(TEST_MASK_DIR,MASK_FILES[IMG_ID]),allow_pickle=True).item()

    df = texture_single_values(channel,test_mask[channel],test_img[channel])
    # print(df)
    expected_df = pd.read_csv(os.path.join(TEST_CSV_DIR,'texture_single_values_output.csv'))
    pd.testing.assert_frame_equal(df,expected_df)

# @pytest.mark.skip(reason="not implemented yet")
def test_granularity():
    channel = 'w3'
    test_img = np.load(os.path.join(TEST_IMG_DIR,IMG_FILES[IMG_ID]),allow_pickle=True).item()

    df = granularity(channel,test_img[channel])
    expected_df = pd.read_csv(os.path.join(TEST_CSV_DIR,'granularity_output.csv'))
    pd.testing.assert_frame_equal(df,expected_df)

def test_intensity():
    channel = 'w3'
    test_img = np.load(os.path.join(TEST_IMG_DIR,IMG_FILES[IMG_ID]),allow_pickle=True).item()
    test_mask = np.load(os.path.join(TEST_MASK_DIR,MASK_FILES[IMG_ID]),allow_pickle=True).item()
    regions = skimage.measure.regionprops(test_mask[channel])
    df = intensity(test_img[channel],test_mask[channel],channel,regions)
    expected_df = pd.read_csv(os.path.join(TEST_CSV_DIR,'intensity_output.csv'))
    pd.testing.assert_frame_equal(df,expected_df)

@pytest.mark.skip(reason="helper function for concentric_measurements()")
def test_create_concentric_areas():
    pass

# @pytest.mark.skip(reason="not implemented yet")
def test_concentric_measurements():
    channel = 'w3'
    test_img = np.load(os.path.join(TEST_IMG_DIR,IMG_FILES[IMG_ID]),allow_pickle=True).item()
    test_mask = np.load(os.path.join(TEST_MASK_DIR,MASK_FILES[IMG_ID]),allow_pickle=True).item()
    df = concentric_measurements(scale=8,ROI=30, simg=test_img[channel],
                                segmented_labels=test_mask[channel], chan=channel,DAPI=0)

    expected_df = pd.read_csv(os.path.join(TEST_CSV_DIR,'concentric_measurements_output.csv'))
    df = df.astype(expected_df.dtypes.to_dict())
    pd.testing.assert_frame_equal(df,expected_df)
@pytest.mark.skip(reason="no way to test this function")
def test_show_cells():
    pass

def test_zernike_measurements():
    channel = 'w3'
    # test_img = np.load(os.path.join(TEST_IMG_DIR,IMG_FILES[IMG_ID]),allow_pickle=True).item()
    test_mask = np.load(os.path.join(TEST_MASK_DIR,MASK_FILES[IMG_ID]),allow_pickle=True).item()
    df = zernike_measurements(test_mask[channel],roi=30,chan=channel)
    expected_df = pd.read_csv(os.path.join(TEST_CSV_DIR,'zernike_measurements_output.csv'))
    df = df.astype(expected_df.dtypes.to_dict())
    pd.testing.assert_frame_equal(df,expected_df)
# @pytest.mark.skip(reason="skeletonize error!")
def test_mitochondria_measurement():
    channel = 'w5'
    test_img = np.load(os.path.join(TEST_IMG_DIR,IMG_FILES[IMG_ID]),allow_pickle=True).item()
    test_mask = np.load(os.path.join(TEST_MASK_DIR,MASK_FILES[IMG_ID]),allow_pickle=True).item()
    
    df = mitochondria_measurement(test_mask[channel],test_img[channel],viz=False)
    expected_df = pd.read_csv(os.path.join(TEST_CSV_DIR,'mitochondria_measurement_output.csv'))
    df = df.astype(expected_df.dtypes.to_dict())
    pd.testing.assert_frame_equal(df,expected_df)


# @pytest.mark.skip(reason="not implemented yet")
def test_RNA_measurement():
    channel = 'w4'
    test_img = np.load(os.path.join(TEST_IMG_DIR,IMG_FILES[IMG_ID]),allow_pickle=True).item()
    test_mask = np.load(os.path.join(TEST_MASK_DIR,MASK_FILES[IMG_ID]),allow_pickle=True).item()

    df = RNA_measurement(test_mask[channel].astype(int),test_img[channel],viz=False)
    expected_df = pd.read_csv(os.path.join(TEST_CSV_DIR,'RNA_measurement_output.csv'))
    df = df.astype(expected_df.dtypes.to_dict())
    pd.testing.assert_frame_equal(df,expected_df)

@pytest.mark.skip(reason="under development")
def test_neuritis_measurement():
    pass

# @pytest.mark.skip(reason="not implemented yet")
def test_correlation_measurements():
    channels = ('w2','w4')
    test_img = np.load(os.path.join(TEST_IMG_DIR,IMG_FILES[IMG_ID]),allow_pickle=True).item()
    test_mask = np.load(os.path.join(TEST_MASK_DIR,MASK_FILES[IMG_ID]),allow_pickle=True).item()

    df = correlation_measurements(test_img[channels[0]],test_img[channels[1]],
                                  channels[0],channels[1],test_mask[channels[0]],
                                test_mask[channels[1]])

    expected_df = pd.read_csv(os.path.join(TEST_CSV_DIR,'correlation_measurements_output.csv'))
    df = df.astype(expected_df.dtypes.to_dict())
    pd.testing.assert_frame_equal(df,expected_df)

# @pytest.mark.skip(reason="broken")
def test_single_cell_feature_extraction():
    test_img = np.load(os.path.join(TEST_IMG_DIR,IMG_FILES[IMG_ID]),allow_pickle=True).item()
    test_img_stack = np.stack([test_img['w1'],test_img['w2'],test_img['w3'],test_img['w4'],test_img['w5'],test_img['w6']],axis=0)
    channels = ('w1','w2','w3','w4','w5','w6')
    q_flag, df = single_cell_feature_extraction(test_img_stack.astype(np.uint8),channels,roi=30,mito_ch='w5',rna_ch='w4',
                                        neuritis_ch=None,downsampling=1,viz=False)

    expected_df = pd.read_csv(os.path.join(TEST_CSV_DIR,'single_cell_feature_extraction_output.csv'))
    df = df.astype(expected_df.dtypes.to_dict())

    assert df.isna().values.any() == False
    assert np.isinf(df).values.sum() == 0
    pd.testing.assert_frame_equal(df.reset_index(drop=True),expected_df.reset_index(drop=True))
