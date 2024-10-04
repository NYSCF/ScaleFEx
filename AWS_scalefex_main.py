import yaml
import os
import sched
import time
import logging
import data_query.query_functions_AWS as dq

# Define the root directory
ROOT_DIR = '/'.join(__file__.split('/')[:-1])

# Ensure the outputs folder exists
output_dir = os.path.join(ROOT_DIR, 'outputs')
os.makedirs(output_dir, exist_ok=True)
# Setup logging
log_file_path = os.path.join(output_dir, f'screen_init.log')
logging.basicConfig(
    filename=log_file_path,  # Log file name in the outputs folder
    filemode='w',            # Overwrite the log file at the beginning
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    level=logging.INFO)

class Screen_Init: 
    """
    Class representing the initialization of AWS screen.
    
    Methods:
        __init__(yaml_path='parameters.yaml'): 
            Initializes the Screen_Compute object with parameters from a YAML file.
    """
    def __init__(self, yaml_path='try_AWSparameters.yaml'):
        """
        Initializes the Screen_Compute object with parameters from a YAML file.

        Args:
            yaml_path (str): Path to the YAML file containing parameters. Default is 'parameters.yaml'.
        """
        try:
            # Read the yaml file
            with open(yaml_path, 'rb') as f:
                self.parameters = yaml.load(f.read(), Loader=yaml.CLoader)
            exp_name = self.parameters['experiment_name']
            
            logging.info(f"YAML file loaded successfully with parameters: {self.parameters}.")
            # Query data
            logging.info("Querying data with the following parameters: "
                         f"pattern={self.parameters['pattern']}, "
                         f"plate_identifiers={self.parameters['plate_identifiers']}, "
                         f"exp_folder={self.parameters['exp_folder']}, "
                         f"exts={self.parameters['exts']}, "
                         f"experiment_name={self.parameters['experiment_name']}, "
                         f"plates={self.parameters['plates']}, "
                         f"s3_bucket={self.parameters['s3_bucket']}.")
                         
            files, plates = dq.query_data(
                self.parameters['pattern'],
                plate_identifiers=self.parameters['plate_identifiers'], 
                exp_folder=self.parameters['exp_folder'], 
                exts=self.parameters['exts'], 
                experiment_name=self.parameters['experiment_name'],
                plates=self.parameters['plates'], 
                s3_bucket=self.parameters['s3_bucket']
            )
            logging.info(f"Data query completed. Number of files found: {len(files)}.")
            
            # Perform Flat Field Correction (FFC)
            logging.info("Checking if Flat Field Correction (FFC) is needed with FFC parameter: "
                         f"{self.parameters['FFC']}.")
            self.flat_field_correction = {}
        
            if self.parameters['FFC'] is True:
                logging.info("FFC is enabled.")
                ffc_file = os.path.join(self.parameters['experiment_name'] + '_FFC.p')
                logging.info(f"Checking if FFC file exists in S3 with path: {ffc_file}.")
                if not dq.check_s3_file_exists_with_prefix(
                    self.parameters['s3_bucket'], 
                    self.parameters['saving_folder'],
                    self.parameters['exp_folder'],
                    self.parameters['experiment_name']
                ):
                    logging.info(f"{ffc_file} not found, generating FFC with channel: "
                                 f"{self.parameters['channel']} and number of images: "
                                 f"{self.parameters['FFC_n_images']}.")
                    self.flat_field_correction = dq.flat_field_correction_AWS(
                        files, 
                        ffc_file,
                        self.parameters['s3_bucket'],
                        self.parameters['saving_folder'],
                        self.parameters['channel'],
                        self.parameters['experiment_name'],
                        n_images=self.parameters['FFC_n_images']
                    )
                    logging.info("FFC generation completed.")
                else:
                    logging.info(f"{ffc_file} found, using existing FFC.")
            
            logging.info("Uploading FFC to S3 with bucket: "
                         f"{self.parameters['s3_bucket']} and experiment name: "
                         f"{self.parameters['experiment_name']}.")
            dq.upload_ffc_to_s3(self.parameters['s3_bucket'],self.parameters['saving_folder'], 'parameters.yaml', self.parameters['experiment_name'])
            logging.info("FFC upload completed.")

            if len(files) != 0:
                logging.info("Files found, launching EC2 instances with the following parameters: "
                             f"experiment_name={self.parameters['experiment_name']}, "
                             f"region={self.parameters['region']}, "
                             f"s3_bucket={self.parameters['s3_bucket']}, "
                             f"saving_folder={self.parameters['saving_folder']}, "
                             f"amazon_image_id={self.parameters['amazon_image_id']}, "
                             f"instance_type={self.parameters['instance_type']}, "
                             f"nb_subsets={self.parameters['nb_subsets']}, "
                             f"subset_index={self.parameters['subset_index']}, "
                             f"csv_coordinates={self.parameters['csv_coordinates']}, "
                             f"security_group_id={self.parameters['security_group_id']}.")

                instance_ids, instance_tags = dq.launch_ec2_instances(
                    self.parameters['experiment_name'],
                    self.parameters['region'],
                    self.parameters['s3_bucket'],
                    self.parameters['saving_folder'],
                    self.parameters['amazon_image_id'], 
                    self.parameters['instance_type'], 
                    plates, 
                    self.parameters['nb_subsets'], 
                    self.parameters['subset_index'],
                    self.parameters['csv_coordinates'],
                    ScaleFExSubnetA=self.parameters['ScaleFExSubnetA'],
                    ScaleFExSubnetB=self.parameters['ScaleFExSubnetB'],
                    ScaleFExSubnetC=self.parameters['ScaleFExSubnetC'],
                    security_group_id=self.parameters['security_group_id']
                )
                logging.info(f"EC2 instances launched successfully with instance IDs: {instance_ids}.")

                logging.info("Starting periodic check scheduler with interval of 300 seconds.")
                dq.push_log_file(self.parameters['s3_bucket'],self.parameters['saving_folder'],self.parameters['experiment_name'])
                scheduler = sched.scheduler(time.time, time.sleep)
                scheduler.enter(300, 1, dq.periodic_check, (
                    scheduler, 360, dq.check_instance_metrics, 
                    (instance_ids, instance_tags, self.parameters['region'], 0.35, 0)
                ))
                scheduler.run()
                logging.info("Scheduler run completed.")
            
            else:
                logging.info("No files found, skipping EC2 instance launch.")
        
        except Exception as e:
            logging.error(f"An error occurred: {e}", exc_info=True)

def import_module(module_name):
    try:
        logging.info(f"Importing module '{module_name}'.")
        module = __import__(module_name)
        logging.info(f"Module '{module_name}' imported successfully.")
        return module
    except ImportError:
        logging.error(f"Module '{module_name}' not found.")
        return None

def main():
    try:
        logging.info("Main function started.")
        Screen_Init()
        logging.info("Main function completed successfully.")
    except Exception as e:
        logging.error(f"An error occurred in the main function: {e}", exc_info=True)

if __name__ == "__main__":
    main()  # Your main execution block
