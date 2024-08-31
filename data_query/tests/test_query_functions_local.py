'''Tests for query_functions_local.py'''
import pytest
import numpy as np
import pandas as pd
import cv2 as cv
import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(ROOT_DIR)
import query_functions_local

def test_query_data():
    exp_folder = os.path.join(os.path.dirname(__file__),'sample_data','rxrx2','images','HUVEC-1')
    # plate_identifier='Plate'
    plates = [1,2,902342,23,'plates']
    pattern = '<Well>_<Site>_<Channel>.<ext>'
    query_results = query_functions_local.query_data(exp_folder,pattern,plate_identifiers=('Plate',''),plates=plates,exts=('png',))
    expected_output = pd.read_csv( os.path.join(os.path.dirname(__file__),'sample_data','rxrx2','test_query_data_expected_output.csv'))
    assert len(query_results) == 16
    assert query_results.isna().values.any() == False
    assert query_results['plate'].unique().tolist() == ['1','2']
    assert set(query_results.columns) == set(expected_output.columns)-set({'cell_id'})

def test_make_well_and_field_lists():
    csv_file =  os.path.join(os.path.dirname(__file__), 'sample_data', 'rxrx2', 'test_query_data_expected_output.csv')
    files = pd.read_csv(csv_file,index_col=False)
    (wells,sites) = query_functions_local.make_well_and_field_list(files)
    assert list(wells) == ['AB08', 'L18']
    assert list(sites) == ['s1', 's2', 's3', 's4']

# @pytest.mark.skip(reason='function changed')
def test_check_if_file_exists():
    csv_file = os.path.join(os.path.dirname(__file__), 'sample_data', 'rxrx2', 'test_query_data_expected_output.csv')

    coords_file = os.path.join(os.path.dirname(__file__),'sample_data','rxrx2','test_coords.csv')
    # wells = query_functions_local.check_if_file_exists(csv_file,wells= np.array(['AB08', 'L18']),last_field='s5')
    
    _,wells = query_functions_local.check_if_file_exists(csv_file,wells= np.array(['AB08', 'L18']),
                                                       fields=np.array(['s1', 's2', 's3', 's4']))
    assert wells == ['Over']

    _,wells = query_functions_local.check_if_file_exists(csv_file,wells= np.array(['L18']),
                                                       fields=np.array(['s1', 's2', 's3']))
    assert wells == ['L18']
    _,wells = query_functions_local.check_if_file_exists(csv_file,wells= np.array(['L18']),
                                                       fields=np.array(['s1', 's2', 's3','s4','s5']))
    assert wells == ['L18']
    _,coords = query_functions_local.check_if_file_exists(csv_file,wells= np.array(['L18']),
                                                       fields=np.array(['s1', 's2', 's3','s4']),
                                                       coordinates=coords_file,plate=1)

    assert type(coords) is pd.DataFrame
    assert not coords.isna().values.all()
    assert len(coords) == 72

def test_load_image(): 

    img = query_functions_local.load_image(os.path.join(os.path.dirname(__file__),'sample_data','rxrx2','images','HUVEC-1','Plate1','AB08_s1_w1.png'))
    np.testing.assert_equal(img,np.zeros((10,10)).astype(np.uint8))

def test_flat_field_correction_on_data():
    img_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sample_data', 'zstack')

    ffc_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sample_data', 'ffc', 'ffc.npy')

    img_fnames = [f for f in os.listdir(img_dir) if f.endswith('ch1sk1fk1fl1.tiff')] #changed the endswith bc i added more images for bf
    img_paths = [os.path.join(img_dir,f) for f in os.listdir(img_dir) if f.endswith('ch1sk1fk1fl1.tiff')]
    files = pd.DataFrame(columns=['filename'])
    files['filename'] = img_fnames
    files['file_path'] = img_paths
    files['channel'] = [fname.split('-')[-1][:3] for fname in img_fnames]
    print(files)
    ffc_dict = query_functions_local.flat_field_correction_on_data(files,channel=['ch1'], bf_channel = '', n_images=4)
    # np.save(ffc_file,ffc_dict,allow_pickle=True)
    expected_ffc_dict = np.load(ffc_file,allow_pickle=True).item()
    assert isinstance(ffc_dict,dict)
    np.testing.assert_equal(ffc_dict['ch1'],expected_ffc_dict['ch1'])

def test_flat_field_correction_on_data_bf():
    img_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sample_data', 'zstack')
    ffc_file = os.path.join(os.path.dirname(__file__),'sample_data','ffc','bf-ffc.npy')
    img_fnames = [f for f in os.listdir(img_dir) if f.endswith('ch2sk1fk1fl1.tiff')]
    img_paths = [os.path.join(img_dir,f) for f in os.listdir(img_dir) if f.endswith('ch2sk1fk1fl1.tiff')]
    files = pd.DataFrame(columns=['filename'])
    files['filename'] = img_fnames
    files['file_path'] = img_paths
    files['channel'] = ['ch2' for fname in img_fnames]
    print(files)
    ffc_dict = query_functions_local.flat_field_correction_on_data(files, channel=['ch2'], bf_channel = 'ch2', n_images=4)
    # np.save('/home/intern/data_query/tests/sample_data/ffc/bf-ffc.npy', ffc_dict, allow_pickle=True)
    expected_ffc_dict = np.load(ffc_file, allow_pickle=True).item()
    assert isinstance(ffc_dict,dict)
    np.testing.assert_allclose(ffc_dict['ch2'],expected_ffc_dict['ch2'], rtol = .5)

def test_process_zstack():
    img_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sample_data', 'zstack')
    img_fnames = [os.path.join(img_dir,f) for f in os.listdir(img_dir) if f.endswith('ch1sk1fk1fl1.tiff')] #changed the endswith bc i added more images 
    expected_output = cv.imread(os.path.join(img_dir,'max_proj.tiff'),cv.IMREAD_GRAYSCALE)
    max_proj = query_functions_local.process_zstack(img_fnames)

    assert max_proj is not None
    assert np.isnan(max_proj).sum() == 0
    # np.testing.assert_equal(max_proj,expected_output)


class TestLoadAndPreprocess:
    ffc = {'ch1':np.ones((540,540)).astype(np.uint8)}
    task_files = pd.read_csv(os.path.join(ROOT_DIR,'tests','sample_data','zstack','task_files.csv')).sort_values(by='file_path').reset_index(drop=True)
    task_files['file_path'] = task_files['file_path'].apply(lambda x: os.path.join(ROOT_DIR,x))

    def test_load_and_preprocess_wrong_img_size(self):
        params = {'task_files':self.task_files,
                    'channels':['ch1'],
                    'well':'r11c06',
                    'site':'f23',
                    'zstack':True,
                    'img_size':(540,540),
                    'flat_field_correction':self.ffc,
                    'downsampling':2,
                    'return_original':True
                    }
        params.update({'img_size':(100,100)})
        np_images, original_images, current_file = query_functions_local.load_and_preprocess(**params)
        assert np_images is None
        assert original_images is None

    def test_load_and_preprocess_zstack(self):
        params = {'task_files':self.task_files,
                'channels':['ch1'],
                'well':'r11c06',
                'site':'f23',
                'zstack':True,
                'img_size':(540,540),
                'flat_field_correction':self.ffc,
                'downsampling':2,
                'return_original':True
                }
        np_images, original_images, current_file = query_functions_local.load_and_preprocess(**params)
        expected_original_img = query_functions_local.load_image(os.path.join(ROOT_DIR,'tests','sample_data','zstack','max_proj.tiff'))
        expected_output_img = query_functions_local.load_image(os.path.join(ROOT_DIR,'tests','sample_data','zstack','zstack_downsampled_output.tiff'))
        assert np_images.shape == (1,270,270,1)
        # assert np.array_equal(original_images[0],expected_original_img)
        # assert np.array_equal(np_images[0,:,:,0],expected_output_img)
        assert original_images[0].shape == (540,540)
    
    def test_load_and_preprocess_no_zstack(self):
        params = {'task_files':self.task_files,
                'channels':['ch1'],
                'well':'r11c06',
                'site':'f23',
                'zstack':False,
                'img_size':(540,540),
                'flat_field_correction':self.ffc,
                'downsampling':2,
                'return_original':True
                }
        np_images, original_images, current_file = query_functions_local.load_and_preprocess(**params)
        expected_original_img = query_functions_local.load_image(os.path.join(ROOT_DIR,'tests','sample_data','zstack','r11c06f23p01-ch1sk1fk1fl1.tiff'))
        expected_output_img = query_functions_local.load_image(os.path.join(ROOT_DIR,'tests','sample_data','zstack','no_zstack_downsampled_output.tiff'))
        assert np.array_equal(original_images[0],expected_original_img)
        # assert np.array_equal(np_images[0,:,:,0],expected_output_img)
        assert np_images.shape == (1,270,270,1)
        assert original_images[0].shape == (540,540)


def test_scale_images():
    img_path = os.path.join(os.path.dirname(__file__), 'sample_data', 'zstack', 'max_proj.tiff')

    img = cv.imread(img_path,cv.IMREAD_UNCHANGED)
    img_size = img.shape
    ds_factor = 2
    ds_img, ds_size = query_functions_local.scale_images(ds_factor, img, img_size)
    expected_output_img = cv.imread(os.path.join(os.path.dirname(__file__),'sample_data','zstack','max_proj_ds.tiff'),cv.IMREAD_UNCHANGED)
    print(ds_img,expected_output_img)
    print(type(ds_img),type(expected_output_img))
    print(np.where((ds_img-expected_output_img)!=0))
    assert np.array_equal(ds_img,expected_output_img)
    assert ds_img.shape == (int(img_size[0]/ds_factor),int(img_size[1]/ds_factor))
    assert tuple(ds_size) == (int(img_size[0]/ds_factor),int(img_size[1]/ds_factor))


test_scale_images()