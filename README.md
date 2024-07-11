## ScaleFEx℠: Feature Extraction for High-Content Imaging

ScaleFEx℠ (Scalable Feature Extraction ) is an open-source Python pipeline designed to extract biologically meaningful features from large high-content imaging (HCI) datasets. By leveraging artificial intelligence, ScaleFEx℠ enhances the analysis of morphological profiles, particularly in distinguishing disease phenotypes that are indiscernible to the human eye.

### Key Features
- **Robust Feature Extraction**: Utilizes advanced algorithms to distill complex image data into critical features that drive insights into cellular phenotypes.
- **High-Content Imaging Focus**: Tailored for large-scale HCI screens, addressing challenges of scale, variability, and high-dimensionality in biomedical imaging.

### Installation [To be changed]

Clone the repository:
```
git clone https://github.com/yourusername/ScaleFEx.git
```

Install dependencies:
```
pip install -r requirements.txt
```

### Usage

1. **Set Parameters**: Configure `parameters.yaml` with your dataset paths and processing preferences.
    - **vector_type**: Write 'scalefex' for the feature vector, '' if you want only the preprocessing part (specified below)
    - **resource**: 'local' for local computation, 'AWS' for cloud computing
    - **n_of_workers**: 60 ;int, n of workers to use in parallel. If computing on AWS, this parameter will be ignored, as it is fixed in the AWS framework
    - **exp_folder**: '/path/to/images/' ;
    - **experiment_name**: 'exp001' ;this name will be appended to the saved files
    - **saving_folder**: '/path/to/saving/folder/' ;path to the saving folder
    - **plates**: ['1','2'] ;if you want to process a subset of plates, 'all' for all of the ones found in the folder
    - **plate_identifiers**: ['Plate','_'] ;identifier for finding the plate number; should directly precede and follow the plate identifier (eg exp_folder/Plate1/)
    - **pattern**: 'Images/<Well>f<Site>p<Plane(2)>-<Channel(3)>.<ext>' # pattern of the image file: specify all the characters that make up the filepath indicating the location (more details in the wiki) 
    - **file_extensions**: ['tiff',] ; specify the extensions of the image files 
    - **image_size**: [2160,2160] ;size of the image
    - **channel**: ['ch4','ch1', 'ch5',  'ch3', 'ch2'] ;channels to be processed. NOTE: the nuclear channel should be first
    - **zstack**: False ;Set to True if you have multi-plane images
    - **ROI**: 150 ;Radius of the crop to be cut around the cell
    - **neurite_tracing**: '' ;channel where to compute tracing (if any) 
    - **RNA_channel**: 'ch5' ;RNA channel (if any) #set only if you want to compute ScaleFex
    - **Mito_channel**: 'ch2' ;Mitochpndria channel (if any) #set only if you want to compute ScaleFex
    - **downsampling**: 1 ;Downsampling ratio
    - **QC**: True ;True if the user wants to compute a tile-level Quality Control step
    - **FFC**: True ;True to compute the Flat Field Correction
    - **FFC_n_images**: 500 ; n of images to be used to produce the background trend image for the Flat Field Correction
    - **csv_coordinates**: '' ; '' if you don't want to use a pre-computed coordinates file, otherwise, path to the coordinate file. The columns and format of the csv file needs to be as follows, which is the output of the pipeline: <img width="496" alt="image" src="https://github.com/NYSCF/NYSCF_HCI_image_processing/assets/23292813/e25a6268-60e6-4297-9532-a20d4c373e21">
    
    If another code is used to extract the coordinates and the information about the distance is missing, make an empty column called distance
    - **segmenting_function**: 'Nuclei_segmentation.nuclei_location_extraction' for threholding method
    - **save_coordinates**: True ; save a csv file with the coordinates for each plate
    - **min_cell_size**: 200 ;min area of the cell, all the object with smaller area will be removed
    - **max_cell_size**: 100000 ;max area of the cell, all the object with bigger area will be removed
    - **visualization**: False ; if true, the segmentation masks of the entire field will be visualizaed (using matplotlib). NOTE: we suggest to visualize the masks for testing, but to turn it off during the processing of large screens

    - **visualize_masks**: False ; visualize the segmentation mask from each channel. NOTE: we suggest to visualize the masks for testing, but to turn it off during the processing of large screens
    - **visualize_crops**: False ; visualizes the crop of the cell. This helps setting the best ROI size, but we suggest to visualize the crop for testing, but to turn it off during the processing of large screens
    
   **AWS parameters**  GAB TO DESCRIBE BRIEFLY
   - **s3_bucket**: 'nyscf-scalefex'
   - **nb_subsets**: 6 #How many machines per plate
   - **subset_index**: 1 #Which subset is being computed
   - **region**: 'us-east-1'
   - **instance_type**: 'c5.12xlarge' # Machine type
   - **amazon_image_id**: 'ami-06c68f701d8090592' # AMI linked to region
   - **ScaleFExSubnetA**: 'subnet-0f55e119abeabb79d'
   - **ScaleFExSubnetB**: 'subnet-0f2fffa1123aba4e7'
   - **ScaleFExSubnetC**: 'subnet-0811845bbe0f6b56e' 
   - **security_group_id**: 'sg-0e85edf74867bae19' 
  
  
      
3. **Execute Analysis**:
    
   If running the code locally:  
   From the terminal:
   After setting the parameters of the yaml file, navigate to the folder of your code and execute
   ```
   python vector_extraction_class.py
   ```
   
### Example

[TO BE DONE]: jupyter notebook with example of a site

### Contributing

We welcome contributions to ScaleFEx℠. ? Please read `CONTRIBUTING.md` for guidance on submitting pull requests.

### License

ScaleFEx℠ is released under the MIT License. [To be updated] See [LICENSE.md](LICENSE.md) for more details.

### Acknowledgements
??? maybe
This project is supported by researchers dedicated to advancing drug discovery and understanding disease mechanisms through image-based cellular profiling.
