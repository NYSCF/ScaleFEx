import os
import boto3
import pandas as pd

def download_files_from_s3(bucket_name, s3_prefix, local_dir):
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix)
    
    # Ensure local directory exists
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    
    local_files = []
    
    # Download all files from the given prefix
    for page in page_iterator:
        if "Contents" in page:
            for obj in page["Contents"]:
                key = obj["Key"]
                if key.endswith('.parquet'):
                    local_path = os.path.join(local_dir, os.path.basename(key))
                    if not os.path.exists(local_path):
                        s3.download_file(bucket_name, key, local_path)
                        print(f"Downloaded {local_path}")
                    else:
                        print(f"File {local_path} already exists, skipping download.")
                    local_files.append(local_path)
    
    return local_files

def read_parquet_file(file):
    try:
        df = pd.read_parquet(file)
        return df
    except Exception as e:
        print(f"Error reading {file}: {e}")
        return None

def aggregate_files(local_files, s3_prefix, output_file_qc, output_file_sf):
    # Extract the last two elements of the s3_prefix
    s3_prefix_parts = s3_prefix.strip('/').split('/')
    prefix_last_two = s3_prefix_parts[-2:]
    
    # Separate QC and SF files
    local_files_qc = [file for file in local_files if file.endswith('.parquet') and os.path.basename(file).startswith('QC') and all(part in os.path.basename(file) for part in prefix_last_two)]
    local_files_sf = [file for file in local_files if file.endswith('.parquet') and os.path.basename(file).startswith('SF') and all(part in os.path.basename(file) for part in prefix_last_two)]
    
    # Aggregate QC parquet files
    if local_files_qc:
        dataframes_qc = [read_parquet_file(file) for file in local_files_qc]
        dataframes_qc = [df for df in dataframes_qc if df is not None]  # Filter out None entries
        if dataframes_qc:
            combined_df_qc = pd.concat(dataframes_qc, ignore_index=True)
            combined_df_qc.to_parquet(output_file_qc)
            print(f"Aggregated QC file saved to {output_file_qc}")
        else:
            print("No valid QC parquet files found to aggregate.")

    # Aggregate SF parquet files
    if local_files_sf:
        dataframes_sf = [read_parquet_file(file) for file in local_files_sf]
        dataframes_sf = [df for df in dataframes_sf if df is not None]  # Filter out None entries
        if dataframes_sf:
            combined_df_sf = pd.concat(dataframes_sf, ignore_index=True)
            combined_df_sf.to_parquet(output_file_sf)
            print(f"Aggregated SF file saved to {output_file_sf}")
        else:
            print("No valid SF parquet files found to aggregate.")

def analyze_aggregated_file(file_path):
    try:
        df = pd.read_parquet(file_path)
        # Columns that might need decoding
        columns_to_decode = ['Well', 'channel', 'batch', 'well','Site','site']
        for col in columns_to_decode:
            if col in df.columns and df[col].dtype == object:
                df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
                df[col] = df[col].astype('object')  # Ensure dtype is 'object'
        print(f"Header of {file_path}:")
        print(df.head())
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

# Launch
bucket_name = 'nyscf-scalefex'
s3_prefix = 'resultfolder/NIHB111/201/'
s3_prefix_parts = s3_prefix.strip('/').split('/')
prefix_last_two = s3_prefix_parts[-2:]
string_last_two = '_'.join(prefix_last_two)
local_dir = 'scalefex'
output_file_qc = f'aggregated_QC_{string_last_two}.parquet'
output_file_sf = f'aggregated_SF_{string_last_two}.parquet'

if not os.path.exists(output_file_qc) or not os.path.exists(output_file_sf):
    local_files = download_files_from_s3(bucket_name, s3_prefix, local_dir)
    aggregate_files(local_files, s3_prefix, output_file_qc, output_file_sf)
else:
    print(f"Aggregated files already exist. Skipping aggregation step.")

# Analyze the aggregated files
analyze_aggregated_file(output_file_qc)
analyze_aggregated_file(output_file_sf)
