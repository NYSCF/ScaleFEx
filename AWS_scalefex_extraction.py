import yaml
import os
import pickle
import scalefex_utils
import pandas as pd
import numpy as np
from datetime import datetime
import data_query.query_functions_AWS as dq
import time
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from multiprocessing import Lock

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
        
        self.files, plates = dq.query_data(
            self.parameters['pattern'],
            plate_identifiers=self.parameters['plate_identifiers'],
            exp_folder=self.parameters['exp_folder'],
            exts=self.parameters['exts'],
            experiment_name=self.parameters['experiment_name'],
            plates=self.parameters['plates'],
            s3_bucket=self.parameters['s3_bucket']
        )

        self.vec_dir = 'scalefex'
        if not os.path.exists(self.vec_dir):
            os.makedirs(self.vec_dir)

        ffc_file = os.path.join(self.vec_dir, self.parameters['experiment_name'] + '_FFC.p')
        self.flat_field_correction = {}
        if self.parameters['FFC'] and os.path.exists(ffc_file):
            print(f"{ffc_file} Found")
        else:
            for channel in self.parameters['channel']:
                self.flat_field_correction[channel] = 1

        self.plate = plates[0]

        if self.parameters['QC']:
            self.csv_fileQC = os.path.join(
                self.vec_dir, f"QC_{self.parameters['experiment_name']}_{self.plate}_{self.parameters['subset_index']}.csv"
            )

        # Initialize the log file
        self.log_file = os.path.join(self.vec_dir, 'cell_count_log.txt')
        with open(self.log_file, 'w') as log:
            log.write("Well,Cell_Count\n")

        # Initialize a lock for writing to the log file
        self.log_lock = Lock()

        self.start_computation(self.plate, self.files)

    def compute_vector(self, well):
        """
        Function that imports the images and extracts the location of cells.
        """
        cell_counter = 0
        csv_file = os.path.join(
            self.vec_dir, f"SF_{self.parameters['experiment_name']}_{self.plate}_{self.parameters['subset_index']}_01.csv"
        )

        sites = np.unique(self.task_files.site)
        sites.sort()

        for site in sites:
            np_images, original_images, current_file = dq.load_and_preprocess(
                self.task_files, self.parameters['channel'], well, site,
                self.parameters['zstack'], self.parameters['image_size'],
                self.flat_field_correction, self.parameters['downsampling'],
                return_original=self.parameters['QC'],
                s3_bucket=self.parameters['s3_bucket']
            )

            if np_images is not None:
                if not self.parameters['csv_coordinates']:
                    center_of_mass = self.segment_crop_images(np_images[0, :, :, 0])
                    center_of_mass = [list(row) + [n] for n, row in enumerate(center_of_mass)]
                    locations = None
                else:
                    locations = self.locations.astype(str)
                    locations['site'] = locations['site'].apply(lambda x: x.zfill(2))
                    site_str = site.zfill(2)
                    locations = locations[(locations.well == well) & (locations.site == site_str)]
                    center_of_mass = np.asarray(locations[['coordX', 'coordY', 'cell_id']])

                live_cells = len(center_of_mass)
                print(f"Site: {site}, Well: {well}, Plate: {self.plate}, Cells found: {live_cells}")

                if self.parameters['QC']:
                    indQC = 0
                    QC_vector, indQC = Quality_control_HCI.compute_global_values.calculateQC(
                        len(center_of_mass), live_cells, 'scalefex', original_images, well,
                        self.plate, site, self.parameters['channel'], indQC,self.parameters['neurite_tracing'])
                    QC_vector['file_path'] = current_file
                    self.csv_fileQC = dq.save_qc_file(QC_vector, self.csv_fileQC)

                cell_counter += self.process_cells(center_of_mass, np_images, well, site, csv_file, locations, current_file)

        with self.log_lock:
            with open(self.log_file, 'a') as log:
                log.write(f"{well},{cell_counter}\n")

    def process_cells(self, center_of_mass, np_images, well, site, csv_file, locations, current_file):
        """
        Processes individual cells and computes their features.

        Args:
            center_of_mass (list): List of cell coordinates.
            np_images (ndarray): Numpy array of images.
            well (str): Well identifier.
            site (str): Site identifier.
            csv_file (str): Path to the CSV file for saving results.
            locations (DataFrame or None): DataFrame with location data if available.

        Returns:
            int: Number of successfully processed cells.
        """
        cell_counter = 0
        expected_shape = (len(self.parameters['channel']), self.parameters['ROI'] * 2, self.parameters['ROI'] * 2, 1)

        for x, y, n in center_of_mass:
            crop = np_images[:, int(float(x) - self.parameters['ROI']):int(float(x) + self.parameters['ROI']),
                             int(float(y) - self.parameters['ROI']):int(float(y) + self.parameters['ROI']), :]

            if crop.shape != expected_shape:
                print(f"Skipping crop with shape {crop.shape} as it does not match the expected shape {expected_shape}.")
                continue

            initial_vector_data = [self.plate, well, site, x, y, n]
            vector = pd.DataFrame(
                np.asarray(initial_vector_data).reshape(1, 6),
                columns=['plate', 'well', 'site', 'coordX', 'coordY', 'cell_id']
            )

            if locations is None:
                tree = KDTree([row[:2] for row in center_of_mass])
                distance, _ = tree.query([x, y], k=2)
                vector['distance'] = distance[1]
            else:
                matching_locs = locations[(locations.coordX == str(x)) & (locations.coordY == str(y))]
                if not matching_locs.empty:
                    vector['distance'] = matching_locs['distance'].values[0]
                else:
                    print(f"No matching location found for x: {x}, y: {y}")
                    vector['distance'] = np.nan

            try:
                scalefex = ScaleFEx_from_crop.compute_ScaleFEx.ScaleFEx(
                    crop, channel=self.parameters['channel'],
                    mito_ch=self.parameters['Mito_channel'],
                    rna_ch=self.parameters['RNA_channel'],
                    downsampling=self.parameters['downsampling'],
                    roi=int(self.parameters['ROI'])
                ).single_cell_vector

                if isinstance(scalefex, pd.DataFrame):
                    vector = pd.concat([vector, scalefex], axis=1)
                    vector['file_path'] = current_file
                    vector['ROI_size'] = self.parameters['ROI']
                    vector['channel_order'] = str(self.parameters['channel'])
                    vector['downsampling'] = self.parameters['downsampling']
                    csv_file = dq.save_csv_file(
                        vector, csv_file, self.parameters['max_file_size'],
                        self.parameters['s3_bucket'], self.parameters['experiment_name'],
                        self.plate, self.parameters['subset_index']
                    )
                    cell_counter += 1
                else:
                    print(f"scalefex is not a DataFrame. Type: {type(scalefex)}")
            except Exception as e:
                print(f"An error occurred during ScaleFEx computation for crop with shape {crop.shape}: {e} in cell number {n}")
                # scalefex = None  # Ensure scalefex is defined even if an error occurs
                # if 'sc_vec' in locals():
                #     print(f"sc_vec: {sc_vec}")

        return cell_counter

    def start_computation(self, plate, files):
        self.task_files = dq.filter_task_files(files, self.parameters['subset_index'], self.parameters['nb_subsets'])

        if self.parameters['csv_coordinates'] and os.path.exists(self.parameters['csv_coordinates']):
            self.locations = pd.read_csv(self.parameters['csv_coordinates']).astype(str)
            self.locations['plate'] = self.locations['plate'].astype(str)
            self.locations['site'] = self.locations['site'].apply(lambda x: x.zfill(2))
            self.locations = self.locations[self.locations.plate == str(plate)]
            self.locations = dq.filter_coord(self.locations, self.task_files)
            wells = np.unique(self.locations.well)
        else:
            wells = np.unique(self.task_files.well)

        function = self.compute_vector
        scalefex_utils.parallelize(wells, function, self.parameters['n_of_workers'], mode='dev')

        dq.push_all_files(self.parameters['s3_bucket'], self.parameters['experiment_name'], plate, self.parameters['subset_index'], self.vec_dir)
        dq.terminate_current_instance()

    def segment_crop_images(self, img_nuc):
        nls = import_module(self.parameters['segmenting_function'])
        img_mask = nls.compute_DNA_mask(img_nuc)
        center_of_mass = nls.retrieve_coordinates(
            img_mask,
            cell_size_min=self.parameters['min_cell_size'] * self.parameters['downsampling'],
            cell_size_max=self.parameters['max_cell_size'] / self.parameters['downsampling']
        )
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
