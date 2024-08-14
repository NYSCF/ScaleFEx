import pytest
import sys
sys.path.append('/'.join(__file__.split('/')[:-2]))
from scalefex_main import *
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
class TestProcess_HighContentImaging_screen():
    pipeline = Process_HighContentImaging_screen(yaml_path='tests/parameters_test.yaml')
    def test_init_goodyaml(self):
        test_pipeline = Process_HighContentImaging_screen(yaml_path='tests/parameters_test.yaml')
        assert test_pipeline is not None
        assert test_pipeline.parameters is not None
        assert test_pipeline.saving_folder is not None
        assert test_pipeline.csv_file is not None

    def test_init_badyaml(self):
        with pytest.raises(FileNotFoundError):
            Process_HighContentImaging_screen(yaml_path='not_a_file.yaml')

    def test_run(self):
        output_dir = 'sample_data/test/output/'
        for r,d,f in os.walk(output_dir):
            for file in f:
                os.remove(os.path.join(r,file))
        self.pipeline.run()
        # assert os.path.exists(os.path.join(output_dir,'scalefex'))
        # assert os.path.exists(os.path.join(output_dir,'QC_analysis'))
        # assert len([f for f in os.listdir(output_dir) if f.endswith('coordinates.csv')]) == 1
        # assert len([f for f in os.listdir(output_dir) if f.endswith('parameters.yaml')]) == 1
        # assert len(os.listdir(os.path.join(output_dir,'QC_analysis'))) == 1
        # assert len(os.listdir(os.path.join(output_dir,'scalefex'))) == 1


    def test_save_csv_file(self):
        new_output_path = 'sample_data/test/sample_scalefex_vector/Dataset01_Plate1_scalefex_new.csv'
        if os.path.exists(new_output_path):
            os.remove(new_output_path)
        test_output_path = 'sample_data/test/sample_scalefex_vector/Dataset01_Plate1_scalefex.csv'
        vector = pd.read_csv(test_output_path)
        csv_file = new_output_path
        self.pipeline.save_csv_file(vector,csv_file)
        assert os.path.exists(csv_file)
        saved_results = pd.read_csv(csv_file)
        assert saved_results.shape == vector.shape


    def test_segment_crop_images(self):
        # load saved output of data_query.query_functions_local.load_and_preprocess()
        np_images = np.load('sample_data/test/preprocessed_images.npy',allow_pickle=True)
        centroids = self.pipeline.segment_crop_images(np_images[0,:,:,0])
        assert centroids.shape[1] == 2
        assert len(centroids)>0

    def test_segment_crop_images_empty(self):
        # load saved output of data_query.query_functions_local.load_and_preprocess()
        np_images = np.load('sample_data/test/preprocessed_images.npy',allow_pickle=True)
        img_nuc = (np.random.rand(np_images[0,:,:,0].shape[0],np_images[0,:,:,0].shape[1])*10).astype(int)
        centroids = self.pipeline.segment_crop_images(img_nuc)
        assert centroids is not None
        assert len(centroids)==0



