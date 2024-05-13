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
        __init__(yaml_path='param_AWS.yaml'): 
            Initializes the Screen_Compute object with param_AWS from a YAML file.
    """
    def __init__(self, yaml_path='param_AWS.yaml'):
        """
        Initializes the Screen_Compute object with param_AWS from a YAML file.

        Args:
            yaml_path (str): Path to the YAML file containing param_AWS. Default is 'param_AWS.yaml'.
        """
        # Read the yaml file
        with open(yaml_path, 'rb') as f:
            self.param_AWS = yaml.load(f.read(), Loader=yaml.CLoader)
        files = dq.query_data(self.param_AWS['fname_pattern'],plate_identifier = self.param_AWS['plate_identifier'],
                            delimiters = self.param_AWS['fname_delimiters'],exts=self.param_AWS['file_extensions'],
                            experiment_name = self.param_AWS['experiment_name'],plates=self.param_AWS['plates'], 
                            s3_bucket = self.param_AWS['s3_bucket'])
        
        # Perform Flat Field Correction (FFC)
        self.flat_field_correction = {}
    
        if self.param_AWS['FFC'] is True:
            ffc_file = os.path.join(self.param_AWS['experiment_name'] + '_FFC.p')
            if not os.path.exists(ffc_file):
                print(ffc_file + ' Not found generating FFC now')
                self.flat_field_correction = dq.flat_field_correction_AWS(files,ffc_file,
                self.param_AWS['s3_bucket'],self.param_AWS['channel'],self.param_AWS['experiment_name'],
                n_images=self.param_AWS['FFC_n_images'])
            else:
                print(ffc_file + ' Found, loading FFC')
                self.flat_field_correction = pickle.load(open(ffc_file, "rb"))
        else:
            for channel in self.param_AWS['channel']:
                self.flat_field_correction[channel] = 1

        dq.upload_ffc_to_s3(self.param_AWS['s3_bucket'],'param_AWS.yaml',self.param_AWS['experiment_name'])

        if len(files) != 0:
            dq.launch_ec2_instances(self.param_AWS['experiment_name'],
                self.param_AWS['image_id'], self.param_AWS['instance_type'], self.param_AWS['plates'], self.param_AWS['nb_subsets'])
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
