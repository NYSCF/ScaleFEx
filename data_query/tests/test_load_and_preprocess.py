# from datetime import datetime
# import cv2 as cv
# import numpy as np
# import pandas as pd
# import os,sys
# import pytest
# ROOT_DIR = '/'.join(__file__.split('/')[:-2])
# sys.path.append(ROOT_DIR)
# from load_and_preprocess import load_and_preprocess,scale_images
# import query_functions_local as data_retrieve_local

# @pytest.mark.skip(reason="Skipping test_load_and_preprocess_zstack")
# class TestLoadAndPreprocess:
#     ffc = {'ch1':np.ones((540,540)).astype(np.uint8)}
#     task_files = pd.read_csv(os.path.join(ROOT_DIR,'tests/sample_data/zstack/task_files.csv')).sort_values(by='file_path').reset_index(drop=True)
#     task_files['file_path'] = task_files['file_path'].apply(lambda x: os.path.join(ROOT_DIR,x))

#     def test_load_and_preprocess_wrong_img_size(self):
#         params = {'task_files':self.task_files,
#                     'channels':['ch1'],
#                     'well':'r11c06',
#                     'site':'f23',
#                     'zstack':True,
#                     'data_retrieve':data_retrieve_local,
#                     'img_size':(540,540),
#                     'flat_field_correction':self.ffc,
#                     'downsampling':2,
#                     'return_original':True
#                     }
#         params.update({'img_size':(100,100)})
#         np_images, original_images = load_and_preprocess(**params)
#         assert np_images is None
#         assert original_images is None

#     def test_load_and_preprocess_zstack(self):
#         params = {'task_files':self.task_files,
#                 'channels':['ch1'],
#                 'well':'r11c06',
#                 'site':'f23',
#                 'zstack':True,
#                 'data_retrieve':data_retrieve_local,
#                 'img_size':(540,540),
#                 'flat_field_correction':self.ffc,
#                 'downsampling':2,
#                 'return_original':True
#                 }
#         np_images, original_images = load_and_preprocess(**params)
#         expected_original_img = data_retrieve_local.load_image(os.path.join(ROOT_DIR,'tests/sample_data/zstack/max_proj.tiff'))
#         expected_output_img = data_retrieve_local.load_image(os.path.join(ROOT_DIR,'tests/sample_data/zstack/zstack_downsampled_output.tiff'))
#         assert np_images.shape == (1,270,270,1)
#         assert np.array_equal(original_images[0],expected_original_img)
#         assert np.array_equal(np_images[0,:,:,0],expected_output_img)
#         assert original_images[0].shape == (540,540)
    
#     def test_load_and_preprocess_no_zstack(self):
#         params = {'task_files':self.task_files,
#                 'channels':['ch1'],
#                 'well':'r11c06',
#                 'site':'f23',
#                 'zstack':False,
#                 'data_retrieve':data_retrieve_local,
#                 'img_size':(540,540),
#                 'flat_field_correction':self.ffc,
#                 'downsampling':2,
#                 'return_original':True
#                 }
#         np_images, original_images = load_and_preprocess(**params)
#         expected_original_img = data_retrieve_local.load_image(os.path.join(ROOT_DIR,'tests/sample_data/zstack/r11c06f23p01-ch1sk1fk1fl1.tiff'))
#         expected_output_img = data_retrieve_local.load_image(os.path.join(ROOT_DIR,'tests/sample_data/zstack/no_zstack_downsampled_output.tiff'))
#         assert np.array_equal(original_images[0],expected_original_img)
#         assert np.array_equal(np_images[0,:,:,0],expected_output_img)
#         assert np_images.shape == (1,270,270,1)
#         assert original_images[0].shape == (540,540)

# @pytest.mark.skip(reason="Skipping test_scale_images")
# def test_scale_images():
#     img_path = os.path.join("/".join(__file__.split('/')[:-1]),'sample_data/zstack/max_proj.tiff')
#     img = cv.imread(img_path,cv.IMREAD_UNCHANGED)
#     img_size = img.shape
#     ds_factor = 2
#     ds_img, ds_size = scale_images(ds_factor, img, img_size)
#     expected_output_img = cv.imread(os.path.join("/".join(__file__.split('/')[:-1]),'sample_data/zstack/max_proj_ds.tiff'),cv.IMREAD_UNCHANGED)
#     assert np.array_equal(ds_img,expected_output_img)
#     assert ds_img.shape == (int(img_size[0]/ds_factor),int(img_size[1]/ds_factor))
#     assert tuple(ds_size) == (int(img_size[0]/ds_factor),int(img_size[1]/ds_factor))