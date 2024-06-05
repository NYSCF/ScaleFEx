import yaml,os,pickle
import pandas as pd
import numpy as np
from datetime import datetime
import data_query.query_functions_AWS as dq
import boto3
import sched
import time

ROOT_DIR = '/'.join(__file__.split('/')[:-1])

class Screen_Init: 
    """
    Class representing the initialisation of AWS screen.

    Methods:
        __init__(yaml_path='AWS_params.yaml'): 
            Initializes the Screen_Compute object with AWS_params from a YAML file.
    """
    def __init__(self, yaml_path='AWS_params.yaml'):
        """
        Initializes the Screen_Compute object with AWS_params from a YAML file.

        Args:
            yaml_path (str): Path to the YAML file containing AWS_params. Default is 'AWS_params.yaml'.
        """
        # Read the yaml file
        with open(yaml_path, 'rb') as f:
            self.AWS_params = yaml.load(f.read(), Loader=yaml.CLoader)
        files = dq.query_data(self.AWS_params['fname_pattern'],plate_identifier = self.AWS_params['plate_identifier'],
                            delimiters = self.AWS_params['fname_delimiters'],exts=self.AWS_params['file_extensions'],
                            experiment_name = self.AWS_params['experiment_name'],plates=self.AWS_params['plates'], 
                            s3_bucket = self.AWS_params['s3_bucket'])
        
        # Perform Flat Field Correction (FFC)
        self.flat_field_correction = {}
    
        if self.AWS_params['FFC'] is True:
            ffc_file = os.path.join(self.AWS_params['experiment_name'] + '_FFC.p')
            if not os.path.exists(ffc_file):
                print(ffc_file + ' Not found generating FFC now')
                self.flat_field_correction = dq.flat_field_correction_AWS(files,ffc_file,
                self.AWS_params['s3_bucket'],self.AWS_params['channel'],self.AWS_params['experiment_name'],
                n_images=self.AWS_params['FFC_n_images'])
            else:
                print(ffc_file + ' Found, loading FFC')
                self.flat_field_correction = pickle.load(open(ffc_file, "rb"))
        else:
            for channel in self.AWS_params['channel']:
                self.flat_field_correction[channel] = 1

        dq.upload_ffc_to_s3(self.AWS_params['s3_bucket'],'AWS_params.yaml',self.AWS_params['experiment_name'])

        if len(files) != 0:
            instance_ids,instance_tags = dq.launch_ec2_instances(self.AWS_params['experiment_name'],self.AWS_params['region'],self.AWS_params['s3_bucket'],
                self.AWS_params['amazon_image_id'], self.AWS_params['instance_type'], self.AWS_params['plates'], self.AWS_params['nb_subsets'],
                self.AWS_params['csv_coordinates'],subnet_id = self.AWS_params['subnet_id'],security_group_id =self.AWS_params['security_group_id'])
            scheduler = sched.scheduler(time.time, time.sleep)
            scheduler.enter(300, 1, dq.periodic_check, (scheduler, 360, dq.check_instance_metrics, (instance_ids,instance_tags,0.35,0)))
            scheduler.run()
        
        else  :
            print('No files found')

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
