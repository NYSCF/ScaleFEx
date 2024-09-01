
import yaml, os, pickle
import scalefex_utils
import pandas as pd
import numpy as np
from datetime import datetime
import data_query.query_functions_AWS as dq
import time
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import cv2 as cv
global ScaleFEx_from_crop
import ScaleFEx_from_crop.compute_ScaleFEx
global Quality_control_HCI 
import Quality_control_HCI.compute_global_values

ROOT_DIR = '/'.join(__file__.split('/')[:-1])

class Process_HighContentImaging_screen_on_AWS:
    """
    Class representing the computation of screen data.

    Methods:
        __init__(yaml_path='parameters.yaml'): 
            Initializes the Screen_Compute object with parameters from a YAML file.
    """
    def __init__(self, yaml_path='parameters.yaml'):
        """
        Initializes the Screen_Compute object with parameters from a YAML file.

        Args:
            yaml_path (str): Path to the YAML file containing parameters. Default is 'parameters.yaml'.
        """
        # Read the yaml file
        with open(yaml_path, 'rb') as f:
            self.parameters = yaml.load(f.read(), Loader=yaml.CLoader)
        
        self.files, plates = dq.query_data(self.parameters['pattern'], plate_identifiers=self.parameters['plate_identifiers'], 
                                           exp_folder=self.parameters['exp_folder'], exts=self.parameters['exts'], 
                                           experiment_name=self.parameters['experiment_name'], plates=self.parameters['plates'], 
                                           s3_bucket=self.parameters['s3_bucket'])
        self.plate = plates[0]
        self.vec_dir = 'outputs'
        if not os.path.exists(self.vec_dir):
            os.makedirs(self.vec_dir)

        self.sites_computed_file = os.path.join(self.vec_dir,str(self.parameters['experiment_name'])+ '_' + str(self.plate) + '_' + str(self.parameters['subset_index']) +'_sites-computed.csv')
        pd.DataFrame(columns=['plate','well','site','subset','file_path',
                    'total_count','computed_count','on_edge_count','fail_count','computed_ids',
                    'on_edge_ids','fail_ids']).to_csv(self.sites_computed_file,index=False)
        
        ffc_file = os.path.join(self.vec_dir, self.parameters['experiment_name'] + '_FFC.p')
        self.flat_field_correction = {}

        if self.parameters['FFC'] is True and os.path.exists(ffc_file):
            print(ffc_file + ' Found')
        else:
            for channel in self.parameters['channel']:
                self.flat_field_correction[channel] = 1
            image_size = cv.imread(self.files.iloc[0]['file_path'],cv.IMREAD_GRAYSCALE).shape
            self.parameters['image_size'] = image_size

        self.parameters['image_size'] = self.flat_field_correction[self.parameters['channel'][0]].shape
        
        if self.parameters['QC'] == True:

            self.csv_fileQC = os.path.join(self.vec_dir, 'QC_' + self.parameters['experiment_name'] + '_' + str(self.plate) + '_' 
                                           + str(self.parameters['subset_index']) + '.csv')
        self.start_computation(self.plate, self.files)

    def compute_vector(self, well):
        ''' Function that imports the images and extracts the location of cells '''

        csv_file = os.path.join(self.vec_dir, 'SF_' + self.parameters['experiment_name'] + '_' + self.plate + '_' 
                                + str(self.parameters['subset_index']) + '_01' + '.csv')
        
        sites = np.unique(self.task_files.site)
        sites.sort()

        for site in sites:
            np_images, original_images, current_file = dq.load_and_preprocess(self.task_files, self.parameters['channel'], well, site, 
                                                                            self.parameters['zstack'], self.parameters['image_size'], 
                                                                            self.flat_field_correction, self.parameters['downsampling'], 
                                                                            return_original=self.parameters['QC'], 
                                                                            s3_bucket=self.parameters['s3_bucket'])
            
            if np_images is not None:
                if self.parameters['csv_coordinates'] == '' or self.parameters['csv_coordinates'] is None:
                    center_of_mass = self.segment_crop_images(np_images[0, :, :, 0])
                    center_of_mass = [list(row) + [n] for n, row in enumerate(center_of_mass)]
                else:
                    locations = self.locations
                    locations = locations.astype(str)  # Ensure all columns are strings
                    # Normalize site values for comparison
                    locations['site'] = locations['site'].apply(lambda x: x.zfill(2))  # Ensure leading zeros
                    site_str = site.zfill(2)  # Ensure site has leading zeros if needed
                    locations = locations.loc[(locations.well == well) & (locations.site == site_str)]
                    center_of_mass = np.asarray(locations[['coordX', 'coordY', 'cell_id']])
                
                print(f"Site: {site}, Well: {well}, Plate: {self.plate}, Cells found: {len(center_of_mass)}")

                if self.parameters['QC'] == True:
                    indQC = 0
                    QC_vector, indQC = Quality_control_HCI.compute_global_values.calculateQC(len(center_of_mass), 
                                                                                            'scalefex', original_images, well, 
                                                                                            self.plate, site, self.parameters['channel'], 
                                                                                            indQC)
                    QC_vector['file_path'] = current_file
                    self.csv_fileQC = dq.save_qc_file(QC_vector, self.csv_fileQC)

                is_computed = np.ones(len(center_of_mass)) * -1
                for index, (x, y, n) in enumerate(center_of_mass):
                    crop = np_images[:, int(float(x) - self.parameters['ROI']):int(float(x) + self.parameters['ROI']), 
                                    int(float(y) - self.parameters['ROI']):int(float(y) + self.parameters['ROI']), :]
                    if crop.shape != (len(self.parameters['channel']), self.parameters['ROI'] * 2, self.parameters['ROI'] * 2, 1):
                        print(f"Crop shape {crop.shape} does not match expected shape, cell on the border")
                        is_computed[index] = 0
                        continue

                    else:
                        ind = 0
                        vector = pd.DataFrame(np.asarray([self.plate, well, site, x, y, n]).reshape(1, 6 ), 
                                            columns=['plate', 'well', 'site', 'coordX', 'coordY', 'cell_id'], index=[ind])
                        if self.parameters['csv_coordinates'] == '' or self.parameters['csv_coordinates'] is None:
                            tree = KDTree([row[:2] for row in center_of_mass])
                            # Query the nearest distance and the index of the nearest point
                            distance, _ = tree.query([x, y], k=2)    
                            vector['distance'] = distance[1] 
                        else:
                            distance_values = locations.loc[(locations.coordX == str(x)) & (locations.coordY == str(y)), 'distance'].values
                            if len(distance_values) > 0:
                                vector['distance'] = distance_values[0]
                            else:
                                print(f"No matching distance found for coordinates ({x}, {y})")
                                vector['distance'] = np.nan

                        try:
                            scalefex = ScaleFEx_from_crop.compute_ScaleFEx.ScaleFEx(crop, channel=self.parameters['channel'], 
                                                                                mito_ch=self.parameters['Mito_channel'], 
                                                                                rna_ch=self.parameters['RNA_channel'], 
                                                                                downsampling=self.parameters['downsampling'], 
                                                                                roi=int(self.parameters['ROI'])).single_cell_vector

                            if isinstance(scalefex, pd.DataFrame):
                                vector = pd.concat([vector, scalefex], axis=1)
                                vector['file_path'] = current_file
                                vector['ROI_size'] = self.parameters['ROI']
                                vector['channel_order'] = str(self.parameters['channel'])
                                vector['downsampling'] = self.parameters['downsampling']
                                csv_file = dq.save_csv_file(vector, csv_file, self.parameters['max_file_size'], 
                                                            self.parameters['s3_bucket'],self.parameters['saving_folder'], self.parameters['experiment_name'], 
                                                            self.plate, self.parameters['subset_index'])
                                is_computed[index] = 1
                                                
                        except ValueError as e:
                            print("An error occurred during ScaleFEx computation:", e)
                            is_computed[index] = -1

                center_of_mass = np.array(center_of_mass)
                computed_ids = tuple(center_of_mass[np.argwhere(is_computed == 1).flatten(),2])
                on_edge_ids = tuple(center_of_mass[np.argwhere(is_computed == 0).flatten(), 2])
                failed_ids = tuple(center_of_mass[np.argwhere(is_computed == -1).flatten(), 2])
  
                file_path = self.task_files[(self.task_files['plate'] == self.plate) & (self.task_files['well'] == well) & (self.task_files['site'] == site) &
                                            (self.task_files['channel'] == self.parameters['channel'][0])]['file_path'].iloc[0]
                
                compute_vec = [[self.plate, well, site, self.parameters['subset_index'], file_path,
                                len(center_of_mass),len(computed_ids), len(on_edge_ids), len(failed_ids), 
                                str(computed_ids), str(on_edge_ids),str(failed_ids)]]
                
                site_row = pd.DataFrame(data=compute_vec, columns=self.fields_computed_df.columns)
                
                site_row.to_csv(self.sites_computed_file, mode='a', header=False, index=False)



    def start_computation(self, plate, files):
        self.task_files = dq.filter_task_files(files, self.parameters['subset_index'], self.parameters['nb_subsets']) 

        self.fields_computed_df = pd.read_csv(self.sites_computed_file,converters={'plate':str,'well':str,'site':str,
                                                                               'computed_ids':str,'on_edge_ids':str,'fail_ids':str})

        if self.parameters['csv_coordinates'] is not None and os.path.exists(self.parameters['csv_coordinates']):
            self.locations = pd.read_csv(self.parameters['csv_coordinates'])
            self.locations = self.locations.astype(str)  # Ensure all columns are strings
            self.locations['plate'] = self.locations['plate'].astype(str)
            self.locations['site'] = self.locations['site'].apply(lambda x: x.zfill(2))  # Ensure leading zeros
            self.locations = self.locations.loc[self.locations.plate == str(plate)]
            self.locations = dq.filter_coord(self.locations, self.task_files)
            wells = np.unique(self.locations.well)
        else:
            wells = np.unique(self.task_files.well)

        function = self.compute_vector
        scalefex_utils.parallelize(wells, function, self.parameters['n_of_workers'], mode='dev')
        
        dq.push_all_files(self.parameters['s3_bucket'], self.parameters['saving_folder'],self.parameters['experiment_name'], plate, self.parameters['subset_index'], self.vec_dir)
        dq.terminate_current_instance()

    def segment_crop_images(self, img_nuc):
        # extraction of the location of the cells
        nls = import_module(self.parameters['segmenting_function'])
        img_mask = nls.compute_DNA_mask(img_nuc)
        center_of_mass = nls.retrieve_coordinates(img_mask, 
                                                  cell_size_min=self.parameters['min_cell_size'] * self.parameters['downsampling'], 
                                                  cell_size_max=self.parameters['max_cell_size'] / self.parameters['downsampling'])
        try:
            print('N of cells found: ', len(center_of_mass))
        except NameError:
            center_of_mass = []
            print('No Cells detected')
        
        return center_of_mass
                 
def import_module(module_name):
    try:
        module = __import__(module_name)
        return module
    except ImportError:
        print(f"Module '{module_name}' not found.")
        return None
    
def main():
    Process_HighContentImaging_screen_on_AWS()

if __name__ == "__main__":
    main()


