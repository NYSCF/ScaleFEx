import yaml,os,pickle
import pandas as pd
import numpy as np
from datetime import datetime
import data_query.query_functions_AWS as dq
import boto3

ROOT_DIR = '/'.join(__file__.split('/')[:-1])

class Screen_Init: #come up with a better name
    """
    Class representing the initialisation of AWS screen.

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

        self.saving_folder = self.parameters['saving_folder']
        files = dq.query_data(self.parameters['fname_pattern'],plate_identifier = self.parameters['plate_identifier'],
                            delimiters = self.parameters['fname_delimiters'],exts=self.parameters['file_extensions'],
                            experiment_name = self.parameters['experiment_name'],plates=self.parameters['plates'], 
                            s3_bucket = self.parameters['s3_bucket'])
        
        # Perform Flat Field Correction (FFC)
        self.flat_field_correction = {}
    
        if self.parameters['FFC'] is True:
            ffc_file = os.path.join(self.parameters['experiment_name'] + '_FFC.p')
            if not os.path.exists(ffc_file):
                print(ffc_file + ' Not found generating FFC now')
                self.flat_field_correction = dq.flat_field_correction_AWS(files,ffc_file,
                self.parameters['s3_bucket'],self.parameters['channel'],self.parameters['experiment_name'],
                n_images=self.parameters['FFC_n_images'])
            else:
                print(ffc_file + ' Found, loading FFC')
                self.flat_field_correction = pickle.load(open(ffc_file, "rb"))
        else:
            for channel in self.parameters['channel']:
                self.flat_field_correction[channel] = 1

        dq.upload_ffc_to_s3(self.parameters['s3_bucket'],'parameters.yaml',self.parameters['experiment_name'])

        if len(files) != 0:
            dq.launch_ec2_instances(self.parameters['experiment_name'],
                self.parameters['image_id'], self.parameters['instance_type'], self.parameters['plates'], self.parameters['nb_subsets'])
        else  :
            print('No files found')

        # dq.terminate_current_instance()

def import_module(module_name):
    try:
        module = __import__(module_name)
        return module
    except ImportError:
        print(f"Module '{module_name}' not found.")
        return None

import cProfile
import pstats

def main():
    Screen_Init()

if __name__ == "__main__":
    pr = cProfile.Profile()
    pr.enable()

    main()  # Your main execution block

    pr.disable()
    with open("profile_results.txt", "w") as f:  # Choose a file path/name
        ps = pstats.Stats(pr, stream=f)
        ps.sort_stats("cumulative")  # Sorting by cumulative time
        ps.print_stats()
