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
        files,plates = dq.query_data(self.parameters['pattern'],plate_identifiers = self.parameters['plate_identifiers'], exp_folder = self.parameters['exp_folder'], 
                              exts=self.parameters['exts'],experiment_name = self.parameters['experiment_name'],
                              plates=self.parameters['plates'], s3_bucket = self.parameters['s3_bucket'])
        
        # Perform Flat Field Correction (FFC)
        self.flat_field_correction = {}
    
        if self.parameters['FFC'] is True:
            ffc_file = os.path.join(self.parameters['experiment_name'] + '_FFC.p')
            if not os.path.exists(ffc_file):
                print(ffc_file + ' Not found generating FFC now')
                self.flat_field_correction = dq.flat_field_correction_AWS(files,ffc_file,
                self.parameters['s3_bucket'],self.parameters['channel'],self.parameters['experiment_name'],n_images=self.parameters['FFC_n_images'])
            else:
                print(ffc_file + ' Found, loading FFC')
                self.flat_field_correction = pickle.load(open(ffc_file, "rb"))
        else:
            for channel in self.parameters['channel']:
                self.flat_field_correction[channel] = 1

        dq.upload_ffc_to_s3(self.parameters['s3_bucket'],'parameters.yaml',self.parameters['experiment_name'])

        if len(files) != 0:
            instance_ids,instance_tags = dq.launch_ec2_instances(self.parameters['experiment_name'],self.parameters['region'],self.parameters['s3_bucket'],
                self.parameters['amazon_image_id'], self.parameters['instance_type'], plates, self.parameters['nb_subsets'],
                self.parameters['csv_coordinates'],subnet_id = self.parameters['subnet_id'],security_group_id =self.parameters['security_group_id'])
            scheduler = sched.scheduler(time.time, time.sleep)
            scheduler.enter(300, 1, dq.periodic_check, (scheduler, 360, dq.check_instance_metrics, (instance_ids,instance_tags,
                                                                                                    self.parameters['region'],0.35,0)))
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
