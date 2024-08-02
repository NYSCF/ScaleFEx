''' Functions for data querying and handling in local machines'''
import glob, os, cv2 ,pickle,sys
import numpy as np
import pandas as pd
import boto3
import time
from io import BytesIO
import io
from PIL import Image
from ec2_metadata import ec2_metadata
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timedelta, timezone
from prettytable import PrettyTable
from botocore.exceptions import ClientError
from .query_functions_local import *


def query_data(pattern,plate_identifiers='',exts=('tiff',), exp_folder = ''
               ,experiment_name = '', plates=['101'], s3_bucket = ''):
    ''' 
    Queries the data from the folders and extracts wells, sites and channels. 
    This is the main function to be changed if the user's has the files 
    arranged in a different way. The output is a dataframe that contains plate, well, 
    site, channel, file name and file path of each image 
    
    Arguments:
        exp_folder: string
            Parent directory where

        pattern: string
            A string specifying locations of metadata as well as length of substrings
            Separate metadata fields are specified with <column_name>, optionally including substring length <column_name(substr_length)>
            Length should be specified if there is no character separating the fields.
            Example: 'Images/<Well(6)><Site(3)><Plane(3)>-<Channel(3)>.<ext>' for phenix data that looks like
                     '/mnt/synology01/HTN0001/HTN0001_CCU384_401__2024-02-17T22_04_34-Measurement 19/Images/r03c03f01p01-ch1sk1fk1fl1.tiff'
        
        plate_type: string
            substring or regtex pattern that needs to be in the plate subdirectory name (direct subdirectory of exp_folder)

        exts: tuple of strings
            list of file extensions to search for (e.g., tiff, png, jpg)

        delimiters: tuple of char/str
            list of characters to split filename on
            Default: (' ','-','_')

    Return:
        files_df: pd.DataFrame
            tabular data with metadata fields divided into columns, one row per file
    '''
    if plate_identifiers == None:
        plate_identifiers = ''

    files_df,plate_list = scan_s3(plates,plate_identifiers,exp_folder,exts,experiment_name, s3_bucket)
    plate_pattern = pattern
        
    files_df['filename'] = [i.split('/')[-1] for i in files_df.file_path]
    # extracting metadata based on pattern
    files_df['filename_metadata'] = files_df['file_path'].apply(lambda x: extract_metadata_from_filename(plate_pattern, os.path.basename(x))) 
    files_df = files_df.join(pd.DataFrame(files_df['filename_metadata'].to_list()))

    # can change this if needed
    default_cols = ['plate','well','site','channel','plane','filename','plate_folder','file_path']
    default_vals = {'plate':'plate01',
                    'well':'well01',
                    'site':'site01',
                    'channel':'channel01',
                    'plane':'plane01',
                    'plate_folder':''}

    found_cols = list(set(files_df.columns).intersection(set(default_cols)))
    not_included = list(set(default_cols)-set(files_df.columns))
    for col in not_included:
        files_df[col] = default_vals[col]
    final_cols = found_cols+not_included
    final_cols.sort(key=lambda x: default_cols.index(x))
    files_df = files_df[final_cols]

    return files_df.convert_dtypes(),plate_list
    
def load_and_preprocess(task_files, channels, well, site, zstack, img_size, flat_field_correction,
                        downsampling, return_original=False, s3_bucket=''):
    np_images = []
    original_images = []
    for ch in channels:
        image_fnames = task_files.loc[(task_files.well == well) & (task_files.site == site) & (task_files.channel == ch), 'file_path'].values

        if not zstack:
            img = read_image_from_s3(s3_bucket, image_fnames[0])
        else:
            img = process_zstack_s3(s3_bucket, image_fnames)

        # Check that the image is of the right format
        if (img is not None) and (img.shape[0] == img_size[0]) and (img.shape[1] == img_size[1]):

            if return_original:
                original_images.append(img)

            img = img / (flat_field_correction[ch] * 1e-8)
            if downsampling != 1:
                img, img_size = scale_images(downsampling, img, img_size)

            img = (img / (np.max(img))) * 255
            np_images.append(img.astype('uint8'))

        else:
            print('Img corrupted')
            return None, None, None

    np_images = np.array(np_images)
    np_images = np.expand_dims(np_images, axis=3)

    return np_images, np.array(original_images), image_fnames[0]


def save_qc_file(QC_vector, csv_fileQC):
    if not os.path.exists(csv_fileQC):
        QC_vector.to_csv(csv_fileQC, header=True)
    else:
        QC_vector.to_csv(csv_fileQC, mode='a', header=False)
    return csv_fileQC

def save_csv_file(vector, csv_file, max_file_size, bucket, experiment_name, plate, index_subset):
    '''
    Save the vector in a CSV file.
    If the file exceeds the max_file_size, convert it to Parquet, upload to S3, and delete the original CSV file.
    '''
    while os.path.exists(get_marker_file(csv_file)):
        print(f"Marker file exists for {csv_file}. Incrementing filename.")
        csv_file = increment_filename(csv_file)

    try:
        if not os.path.exists(csv_file):
            print(f"Creating new file {csv_file}")
            vector.to_csv(csv_file, header=True)
        else:
            if os.stat(csv_file).st_size < max_file_size * 10 ** 6:
                # print(f"Appending to existing file {csv_file}")
                vector.to_csv(csv_file, mode='a', header=False)
            else:
                print(f"File size limit reached for {csv_file}. Processing.")
                new_csv_file = increment_filename(csv_file)
                vector.to_csv(new_csv_file, header=True)
                push_and_delete(csv_file, bucket, experiment_name, plate, index_subset)
                return new_csv_file
    except Exception as e:
        print(f"Failed to save : {e}")

    return csv_file


def increment_filename(csv_file):
    pos = csv_file.find('.csv')
    base = csv_file[:pos]
    number_index = len(base) - 1
    while number_index > 0 and base[number_index].isdigit():
        number_index -= 1
    number_index += 1  # Adjust to point to the start of the number

    if number_index < len(base):
        number = int(base[number_index:])  # Extract the number
        new_number = f"{number + 1:02}"  # Increment and format with two digits
        csv_file = base[:number_index] + new_number + '.csv'
    else:
        # No number found; assume this is the first file of the series
        csv_file = base + '01.csv'
    
    return csv_file

def get_marker_file(csv_file):
    return csv_file + '.done'

def create_marker_file(marker_file):
    with open(marker_file, 'w') as f:
        f.write('')

def push_and_delete(csv_file, bucket, experiment_name, plate, index_subset):
    try:
        marker_file = get_marker_file(csv_file)

        # Create a marker file to indicate the file is being processed
        create_marker_file(marker_file)

        print(f"Reading CSV file {csv_file}")
        df = pd.read_csv(csv_file)
        df = df.applymap(lambda x: str(x).encode('utf-8') if isinstance(x, str) else x)
        
        file_name = os.path.splitext(csv_file)[0]
        output_parquet = file_name + '.parquet'
        pq.write_table(pa.Table.from_pandas(df), output_parquet)
        upload_to_s3(bucket, output_parquet, experiment_name, plate, index_subset)
        os.remove(csv_file)
        os.remove(output_parquet)
        
    except Exception as e:
        print(f"Failed to process {csv_file}: {e}")
        
def upload_to_s3(bucket_name, file, experiment_name, plate, index_subset):
    s3 = boto3.client('s3')
    cleaned_filename = os.path.basename(file).replace("/", "_").replace("_home_ec2-user_project_scalefex_", "")
    s3_path = f'resultfolder/{experiment_name}/{plate}/{index_subset}/' + cleaned_filename

    # Check if file already exists in S3
    if check_s3_file_exists(s3, bucket_name, s3_path):
        # Add a unique character or timestamp to the filename to avoid overwriting
        cleaned_filename = add_unique_suffix(cleaned_filename)
        s3_path = f'resultfolder/{experiment_name}/{plate}/{index_subset}/' + cleaned_filename

    s3.upload_file(file, bucket_name, s3_path)
    print(f"Uploaded {file} to s3://{bucket_name}/{s3_path}")
    
def check_s3_file_exists(s3, bucket_name, s3_path):
    try:
        s3.head_object(Bucket=bucket_name, Key=s3_path)
        return True
    except s3.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False
        else:
            raise

def add_unique_suffix(filename):
    import time
    unique_suffix = time.strftime("%Y%m%d-%H%M%S")
    base, ext = os.path.splitext(filename)
    return f"{base}_{unique_suffix}{ext}"

def scan_s3(plates, plate_identifiers, exp_folder, exts, experiment_name, s3_bucket):
    """
    Queries files from an S3 bucket.
    Args:
    - experiment_name (str): Name of the experiment.
    - s3_bucket (str): Name of the S3 bucket.
    - plates (list or str): List of plate names or 'all' to consider all plates.
    - plate_identifiers (list): List of two strings representing the start and end identifiers for plate names.
    - exp_folder (str): Experiment folder name.
    - exts (list): List of file extensions to filter.

    Returns:
    - DataFrame: DataFrame containing file paths.
    """
    print('Querying from bucket', s3_bucket)
    print('Querying from', plate_identifiers)
    print('Querying experiment', experiment_name)
    print('Querying plates', plates)
    print('Querying extensions', exts)
    print('Querying folder', exp_folder)

    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    prefix = exp_folder

    # Step 1: List all folders inside the experiment folder
    folders = []
    for page in paginator.paginate(Bucket=s3_bucket, Prefix=prefix, Delimiter='/'):
        folders.extend([content.get('Prefix') for content in page.get('CommonPrefixes', [])])

    print(f'Found {len(folders)} folders inside the experiment folder')

    # Step 2: Extract and print plate names from folder names
    matching_plates = []
    for folder in folders:
        print(f"Processing folder: {folder}")
        start_idx = folder.find(plate_identifiers[0])
        if start_idx != -1:
            start_idx += len(plate_identifiers[0])
            end_idx = folder.find(plate_identifiers[1], start_idx)
            if end_idx != -1:
                plate = folder[start_idx:end_idx]
                print(f"Extracted plate: {plate}")
                if plates == 'all' or plate in plates:
                    matching_plates.append((folder, plate))
                    print(f"Matching plate found: {plate} in folder: {folder}")

    print(f'{len(matching_plates)} folders contain plate names:')
    print('Matching plate names:', [plate for _, plate in matching_plates])

    # Step 3: Plate by plate go through every element and add it to a dataframe with the key and plate for columns 
    data = []
    def filter_keys(page):
        for content in page.get('Contents', []):
            key = content['Key']
            if key.endswith(tuple(exts)) and experiment_name in key:
                for folder, plate in matching_plates:
                    full_plate_identifier = f"{plate_identifiers[0]}{plate}{plate_identifiers[1]}"
                    if full_plate_identifier in key:
                        data.append((key, plate))

    for folder, _ in matching_plates:
        for page in paginator.paginate(Bucket=s3_bucket, Prefix=folder):
            filter_keys(page)

    print(len(data), 'files found in prod mode')

    data.sort()
    files_df = pd.DataFrame(data, columns=['file_path', 'plate'])
    unique_plates = files_df['plate'].unique().tolist()
    print('Unique plates:', unique_plates)

    return files_df, unique_plates


def read_image_from_s3(bucket, object_name):
    s3 = boto3.client('s3')
    image_data = BytesIO()
    s3.download_fileobj(bucket, object_name, image_data)
    image_data.seek(0)
    im = cv2.imdecode(np.asarray(bytearray(image_data.read()), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    return im

def process_zstack_s3(bucket_name, image_keys):
    '''Computes the stack's max projection from the image names in an S3 bucket'''
    s3 = boto3.client('s3')
    img = []
    for key in image_keys:
        response = s3.get_object(Bucket=bucket_name, Key=key)
        image_bytes = response['Body'].read()
        image = Image.open(io.BytesIO(image_bytes))
        img.append(np.array(image))
    img = np.max(np.asarray(img), axis=0)
    return img

def flat_field_correction_AWS(files,ffc_file,s3_bucket, Channel, experiment_name, bf_channel='', n_images=20):
    ''' Calculates the background trend of the entire experiment to be used for flat field correction'''
    flat_field_correction = {}

    for ch in Channel:
        B = files.loc[files['channel'] == ch].sample(n_images)
        img = read_image_from_s3(s3_bucket, B.iloc[0]['file_path'])
        
        for i in range(1, n_images):
            new_img = read_image_from_s3(s3_bucket, B.iloc[i]['file_path'])
            img = np.stack([new_img, img], axis=2)
            if ch == bf_channel:
                img = np.mean(img, axis=2)
            else:
                img = np.min(img, axis=2)

        if ch == bf_channel:
            flat_field_correction[ch] = 1/img
        else:
            flat_field_correction[ch] = img

    pickle.dump(flat_field_correction, open(ffc_file, "wb"))
    upload_ffc_to_s3(s3_bucket, ffc_file,experiment_name)
    return flat_field_correction

def upload_ffc_to_s3(bucket_name,file,experiment_name):
    s3 = boto3.client('s3')
    s3.upload_file(file,bucket_name,f'resultfolder/{experiment_name}/'+ str(file).replace("/","_").replace("_home_ec2-user_project_",""))
    print(file + ' uploaded')

def push_all_files(bucket, experiment_name, plate, index_subset, folder_path):
    files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv') or file.endswith('.txt')]
    for file in files:
        try:
            if file.endswith('.csv'):
                # Read the CSV file
                df = pd.read_csv(file)
                df = df.applymap(lambda x: str(x).encode('utf-8') if isinstance(x, str) else x)
                # Create the Parquet file name from the CSV file name
                file_name = os.path.splitext(file)[0]
                output_parquet = file_name + '.parquet'
                pq.write_table(pa.Table.from_pandas(df), output_parquet)
                upload_to_s3(bucket, output_parquet, experiment_name, plate, index_subset)
            elif file.endswith('.txt'):
                # Directly upload the log file
                upload_to_s3(bucket, file, experiment_name, plate, index_subset)
        except Exception as e:
            print(f"Failed to process {file}: {e}")


def terminate_current_instance():

    print('Terminating instance')
    ec2 = boto3.client('ec2',region_name=ec2_metadata.region) #Change the region if necessary
    instance_id = ec2_metadata.instance_id
    time.sleep(3)
    _ = ec2.terminate_instances(
    InstanceIds=[
        instance_id,
    ],)

def filter_task_files(task_files, subset_index, nb_subsets):
    # Convert subset_index to integer
    subset_index = int(subset_index)

    if nb_subsets > 1:
        # Count unique well numbers
        unique_wells = task_files['well'].unique()
        total_wells = len(unique_wells)

        # Divide unique well numbers into subsets
        subset_size = total_wells // nb_subsets
        remaining_wells = total_wells % nb_subsets

        subset_wells = [unique_wells[i:i + subset_size].tolist() for i in range(0, total_wells, subset_size)]

        # Distribute remaining wells among subsets
        for i in range(remaining_wells):
            subset_wells[i].append(unique_wells[-(i + 1)])

        subset_index = subset_index - 1  # Adjust for zero-based indexing

        selected_subset_wells = subset_wells[subset_index]
        filtered_task_files = task_files[task_files['well'].isin(selected_subset_wells)]

    else:
        filtered_task_files = task_files

    return filtered_task_files


def check_s3_file_exists_with_prefix(bucket, exp_folder, experiment_name):
    s3 = boto3.client('s3')
    prefix = f'resultfolder/{exp_folder}{experiment_name}_FFC.p'
    print('Looking for ' + prefix)
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    return 'Contents' in response and len(response['Contents']) > 0

def filter_coord(locations, task_files): 
    filtered_locations = locations[locations['well'].isin(task_files['well'].unique())]
    return filtered_locations

def get_subnet_ids(region):
    ec2 = boto3.client('ec2', region_name=region)
    subnet_names = ["ScaleFExSubnetAId", "ScaleFExSubnetBId", "ScaleFExSubnetCId"]
    subnet_ids = []

    for subnet_name in subnet_names:
        response = ec2.describe_subnets(Filters=[{'Name': 'tag:Name', 'Values': [subnet_name]}])
        if response['Subnets']:
            subnet_ids.append(response['Subnets'][0]['SubnetId'])

    return subnet_ids

def get_security_group_id(region):
    ec2 = boto3.client('ec2', region_name=region)
    
    security_group_name = "ScaleFExSecurityGroup"
    response = ec2.describe_security_groups()
    for sg in response['SecurityGroups']:
        for tag in sg.get('Tags', []):
            if tag['Key'] == 'Name' and tag['Value'] == security_group_name:
                return sg['GroupId']
    security_group_id = response['SecurityGroups'][0]['GroupId']
    return security_group_id

# from botocore.exceptions.ClientError

def launch_ec2_instances(experiment_name, region, s3_bucket, linux_ami, instance_type, plate_list, nb_subsets, subset_index, csv_coordinates,
                          ScaleFExSubnetA=None, ScaleFExSubnetB=None, ScaleFExSubnetC=None, security_group_id=None):
    ec2 = boto3.client('ec2', region_name=region)
    
    if ScaleFExSubnetA is None and security_group_id is None:
        subnet_ids = get_subnet_ids(region)
        security_group_id = get_security_group_id(region)
        print('Security closed')
    else:
        subnet_ids = [ScaleFExSubnetA, ScaleFExSubnetB, ScaleFExSubnetC]

    instance_ids = []
    instance_tags = []

    # Determine the subset indices to launch based on the value of subset_index
    if subset_index == 'all':
        subset_indices = range(1, nb_subsets + 1)
    else:
        subset_indices = [int(subset_index)]

    for plate in plate_list:
        plate = str(plate)
        for subset_idx in subset_indices:
            subset_idx_str = str(subset_idx)
            # Create the UserData script with dynamically assigned variables
            user_data_script = f"""#!/bin/bash
            cd /home/ec2-user
            sudo yum update -y
            sudo yum install python3 python3-pip -y
            sudo yum install -y awscli jq
            sudo yum install git -y
            sudo yum install libglvnd-glx -y
            python3 -m pip install --user virtualenv
            python3 -m virtualenv venv
            source venv/bin/activate
            SECRET_NAME="Gab_Github"
            REGION="{region}"
            SECRET=$(aws secretsmanager get-secret-value --secret-id $SECRET_NAME --region $REGION --query SecretString --output text)
            USERNAME=$(echo "$SECRET" | jq -r 'keys[]')
            TOKEN=$(echo "$SECRET" | jq -r '.[keys[]]')
            echo -e "machine github.com\nlogin $USERNAME\npassword $TOKEN" > ~/.netrc 
            chmod 600 ~/.netrc
            MAIN_REPO_URL="https://github.com/NYSCF/ScaleFEx.git"
            git clone $MAIN_REPO_URL
            cd ScaleFEx
            git submodule update --init --recursive
            aws s3 sync s3://{s3_bucket}/resultfolder/{experiment_name}/ . --exclude '*' --include='{csv_coordinates}'
            aws s3 sync s3://{s3_bucket}/resultfolder/{experiment_name}/ . --exclude '*' --include='{experiment_name}_FFC.p'
            aws s3 sync s3://{s3_bucket}/resultfolder/{experiment_name}/ . --exclude '*' --include='parameters.yaml'
            pip install -r AWS_requirements.txt
            sed -i "s|^plates:.*|plates: ['{plate}']|" parameters.yaml
            sed -i "s|^subset_index:.*|subset_index: {subset_idx_str}|" parameters.yaml
            # Ensure the correct ownership and permissions
            cd ..
            sudo chown -R ec2-user:ec2-user ScaleFEx
            sudo chmod -R 755 ScaleFEx
            # Navigate back to the repository and run the Python script
            cd ScaleFEx
            echo "OK" > Ok.txt
            python3 AWS_scalefex_extraction.py
            """
            launched = False
            for subnet_id in subnet_ids:
                if not subnet_id:
                    continue
                try:
                    instance_params = {
                        'ImageId': linux_ami,
                        'InstanceType': instance_type,
                        'MinCount': 1,
                        'MaxCount': 1,
                        'SecurityGroupIds': [security_group_id],
                        'SubnetId': subnet_id,
                        'UserData': user_data_script,
                        'InstanceMarketOptions': {
                            'MarketType': 'spot',
                            'SpotOptions': {
                                'SpotInstanceType': 'one-time',
                                'InstanceInterruptionBehavior': 'terminate'
                            }},
                        'TagSpecifications': [
                            {
                                'ResourceType': 'instance',
                                'Tags': [
                                    {'Key': 'Name',
                                        'Value': f'{experiment_name}_{plate}_{subset_idx_str}'
                                    }]}],
                        'IamInstanceProfile': {'Name': 'ScaleFExInstanceProfile'}
                    }
                    
                    # Launch the instance
                    response = ec2.run_instances(**instance_params)
                    # Append the new instance ID to the list
                    instance_ids.append(response['Instances'][0]['InstanceId'])
                    instance_tags.append(f'{experiment_name}_{plate}_{subset_idx_str}')
                    print(f'Instance {response["Instances"][0]["InstanceId"]} launched for {plate} - {subset_idx_str} in subnet {subnet_id}')
                    launched = True
                    break
                except ClientError as e:
                    if 'InsufficientInstanceCapacity' in str(e):
                        print(f'Insufficient capacity in subnet {subnet_id}. Trying next subnet...')
                    else:
                        raise
            if not launched:
                print(f'Failed to launch instance for {plate} - {subset_index} in any subnet.')


    return instance_ids, instance_tags

def check_instance_metrics(instance_ids, instance_tags,region, threshold_cpu=0.35, threshold_status_check_failed=0):
    ec2 = boto3.client('ec2', region_name=region)
    cloudwatch = boto3.client('cloudwatch', region_name=region)
    
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(minutes=5) 
    instances_to_relaunch = []
    terminated_count = 0

    table = PrettyTable()
    table.field_names = ["Instance ID", "Tag Name", "CPU Utilization (%)", "CPU Threshold", "Status Check"]

    for idx, instance_id in enumerate(instance_ids):
        instance_tag = instance_tags[idx]  # Retrieve corresponding tag name using index
        instance_state = get_instance_state(ec2, instance_id)
        if instance_state == 'terminated':
            terminated_count += 1
            cpu_utilization = "N/A"
            cpu_status = "Terminated"
            status_check = "N/A"
        else:
            cpu_utilization, cpu_status = check_cpu_utilization(cloudwatch, instance_id, start_time, end_time, threshold_cpu)
            status_check = check_status_check(cloudwatch, instance_id, start_time, end_time, threshold_status_check_failed)
    
        table.add_row([instance_id, instance_tag, cpu_utilization, cpu_status, status_check])

        if cpu_status == "Below Threshold" or status_check == "Failed":
            instances_to_relaunch.append(instance_id)

    print(table)
    if terminated_count == len(instance_ids):
        terminate_current_instance()
    if instances_to_relaunch:
        instances_to_relaunch = list(set(instances_to_relaunch))
        reboot_instances(instances_to_relaunch,region)
    else:
        print("No instances need relaunching.")

def check_cpu_utilization(cloudwatch, instance_id, start_time, end_time, threshold):
    response = cloudwatch.get_metric_statistics(
        Namespace='AWS/EC2',
        MetricName='CPUUtilization',
        Dimensions=[{'Name': 'InstanceId', 'Value': instance_id}],
        StartTime=start_time,
        EndTime=end_time,
        Period=300,
        Statistics=['Average']
    )
    if not response['Datapoints']:
        return "No Data", "No Data"
    average_cpu = response['Datapoints'][0]['Average']
    status = "In range" if average_cpu >= threshold else "Below Threshold"
    return f"{average_cpu}%", status

def check_status_check(cloudwatch, instance_id, start_time, end_time, threshold):
    response = cloudwatch.get_metric_statistics(
        Namespace='AWS/EC2',
        MetricName='StatusCheckFailed',
        Dimensions=[{'Name': 'InstanceId', 'Value': instance_id}],
        StartTime=start_time,
        EndTime=end_time,
        Period=300,
        Statistics=['Sum']
    )
    if not response['Datapoints']:
        return "No Data"
    if response['Datapoints'][0]['Sum'] == threshold:
        return "Passed"
    return "Failed"

def get_instance_state(ec2, instance_id):
    response = ec2.describe_instances(InstanceIds=[instance_id])
    return response['Reservations'][0]['Instances'][0]['State']['Name']

def check_all_instances_terminated(instance_states):
    if all(state == 'terminated' for state in instance_states):
        print("All instances are terminated. Exiting.")
        try :
            terminate_current_instance()
        except :
            sys.exit()

def check_metric(cloudwatch, instance_id, metric_name, statistic, start_time, end_time, threshold, metric_description, unit, condition):
    response = cloudwatch.get_metric_statistics(
        Namespace='AWS/EC2',
        MetricName=metric_name,
        Dimensions=[{'Name': 'InstanceId', 'Value': instance_id}],
        StartTime=start_time,
        EndTime=end_time,
        Period=300, 
        Statistics=[statistic]
    )
    # Handle no data
    if not response['Datapoints']:
        print(f"No data for {metric_description} on instance {instance_id}. May be down or inactive.")
        return False
    
    # Fetch the statistic value and check against the threshold
    value = response['Datapoints'][0][statistic] if response['Datapoints'] else 0
    print(f"Value: {value}, Threshold: {threshold}")
    if (condition == 'cpu' and value < threshold) or (condition == 'check' and value != threshold):
        print(f"Warning: {metric_description} for {instance_id} is not within the threshold at {value} {unit} at {end_time}")
        return True
    else:
        print(f"Instance {instance_id} is within normal {metric_description} limits at {value} {unit} at {end_time}")
        return False

def reboot_instances(instance_ids,region):
    ec2 = boto3.client('ec2', region_name=region)
    ec2.reboot_instances(InstanceIds=instance_ids)
    print(f"Instances {instance_ids} have been requested to reboot.")

def periodic_check(scheduler, interval, function, arguments):
    scheduler.enter(interval, 1, periodic_check, (scheduler, interval, function, arguments))
    function(*arguments)

def download_and_aggregate_parquet_files(bucket_name, s3_prefix, local_dir, output_file):
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix)
    
    # Ensure local directory exists
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    
    local_files = []
    
    for page in page_iterator:
        if "Contents" in page:
            for obj in page["Contents"]:
                key = obj["Key"]
                if key.endswith('.parquet'):
                    local_path = os.path.join(local_dir, os.path.basename(key))
                    s3.download_file(bucket_name, key, local_path)
                    local_files.append(local_path)
    
    # Aggregate parquet files
    dataframes = [pd.read_parquet(file) for file in local_files]
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    # Save aggregated file
    combined_df.to_parquet(output_file)
    
    # Clean up local files
    for file in local_files:
        os.remove(file)
    
    print(f"Aggregated file saved to {output_file}")





