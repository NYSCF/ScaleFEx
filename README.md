![Ubuntu](https://github.com/NYSCF/ScaleFEx/actions/workflows/ubuntu-latest.yml/badge.svg)
![MacOS](https://github.com/NYSCF/ScaleFEx/actions/workflows/macos-latest.yml/badge.svg)
![Windows](https://github.com/NYSCF/ScaleFEx/actions/workflows/windows-latest.yml/badge.svg)
[![Python 3.12](https://img.shields.io/badge/python-3.10%2C3.11%2C3.12-lightgrey)](https://www.python.org/downloads/release/python-312/)


## ScaleFEx℠: Feature Extraction for High-Content Imaging

ScaleFEx℠ (Scalable Feature Extraction ) is an open-source Python pipeline designed to extract biologically meaningful features from large high-content imaging (HCI) datasets. 

Read more about it in the preprint: https://doi.org/10.1101/2023.07.06.547985 

![Fig_1_v2 0](https://github.com/user-attachments/assets/9f4af03f-929a-494e-bac5-6610baf8d7e4)

### Key Features
- **Robust Feature Extraction**: Utilizes advanced algorithms to distill complex image data into critical features that drive insights into cellular phenotypes.
- **High-Content Imaging Focus**: Tailored for large-scale HCI screens, addressing challenges of scale, variability, and high-dimensionality in biomedical imaging.
- **Low overhead**: leveraging parallelizations over cores, it maximizes any computer resources
- **AWS implementation**: to scale up even more. Easy to deploy. Cheaper and faster than the state of the art

## Installation Instructions 
### Basic Requirements:
1.  [Git](https://git-scm.com/downloads) (version control)
       - on MacOS, install the XCode command line tools by running `xcode-select --install` in `terminal`
2.  [Anaconda](https://www.anaconda.com/download) (package manager)
3.  Access to your computer's command line (`terminal` on MacOS, Linux;`cmd` on Windows)


**Follow these steps to set up and run ScaleFEx via command line:**

### 1. Create and Activate a New Environment
Ensure you have Conda installed, then create a new environment:

```
conda create --name ScaleFEx python=3.12
conda activate ScaleFEx
```
### 2. Clone the Repository
Clone the repository and navigate into the main folder:
```
git clone https://github.com/NYSCF/ScaleFEx.git
cd ScaleFEx
```
### 3. Install the Package
Install the repository package:
```
pip install .
```
These steps will set up the environment with all necessary dependencies isolated to ensure everything works smoothly.

### 4. Test:
To check if all the packages are correctly installed, run this command without modifying the parameters files:
 ```
python3 scalefex_main.py
```
You should be able to visualize the detected single cells cells from the data provided with the code

## Usage


1. **Set Parameters**: Configure `parameters.yaml` with your dataset paths and processing preferences.
    - **vector_type**: Write 'scalefex' for the feature vector, '' if you want only the preprocessing part (specified below)
    - **resource**: 'local' for local computation, 'AWS' for cloud computing
    - **n_of_workers**: 60 ;int, n of workers to use in parallel. If computing on AWS, this parameter will be ignored, as it is fixed in the AWS framework
   [ example:    ![image](https://github.com/user-attachments/assets/b10603d7-4162-4851-b444-9596dece8126) ]
    - 🟦 **exp_folder**: '/path/to/images/' ;
    - **experiment_name**: 'exp001' ;this name will be appended to the saved files
    - **saving_folder**: '/path/to/saving/folder/' ;path to the saving folder
    - 🟩 **plates**: ['1','2'] ;if you want to process a subset of plates, 'all' for all of the ones found in the folder
    - 🟥 **plate_identifiers**: ['Plate',''] ;identifier for finding the plate number; should directly precede and follow the plate identifier (eg exp_folder/<u>Plate</u>1/*.tiffs;)
    - 🟧 **pattern**: 'Images/<Well\>f<Site\>p<Plane(2)>-<Channel(3)>.<ext>' # pattern of the image file: specify all the characters that make up the filepath indicating the location ([more details in the wiki](https://github.com/NYSCF/ScaleFEx/wiki/Querying-Data)) 
    - 🟪 **file_extensions**: ['tiff',] ; specify the extensions of the image files 
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

   #### 1.1 NOTE ON QUERYING DATA

   The colored parameters described above are used in parsing data and are used together to search for files in the following way:
   
   ![image](https://github.com/user-attachments/assets/b10603d7-4162-4851-b444-9596dece8126)
   
   where each plate in **plates** and each extension in **file_extensions** is substituted to match all possible combinations.\
   Please consult the [Querying Data wiki](https://github.com/NYSCF/ScaleFEx/wiki/Querying-Data) for more information.

   **AWS parameters**  
   - **s3_bucket**: 'your-bucket'; name of the S3 Bucket storing your images
   - **nb_subsets**: 6; how many machines per plate you want to deploy
   - **subset_index**: 'all'; can use an int to compute a specific subset to compute (i.e:'2')
   - **region**: 'us-east-1'; what region you want to deploy machines into
   - **instance_type**: 'c5.12xlarge' ; Machine type/size
   - **amazon_image_id**: 'ami-06c68f701d8090592' ; AMI linked to region
   - **ScaleFExSubnetA**: 'subnet-0f55e119abeabb79d'; ARN of the subnet you want to use for machines deployment, empty string if you want to use the default one
   - **ScaleFExSubnetB**: 'subnet-0f2fffa1123aba4e7'; second subnet you want to use, if only one use the same
   - **ScaleFExSubnetC**: 'subnet-0811845bbe0f6b56e' ; third subnet you want to use, if only one use the same
   - **security_group_id**: 'sg-0e85edf74867bae19' ; security group you want to use, empty string if you want to use the default one
  

  
      
2. **Execute Analysis**:
    
   If running the code locally:  
   From the terminal:
   After setting the parameters of the yaml file and updating the parameter.yaml file name and location within the scalefex_main.py file, navigate to the folder of your code and execute
   ```
   python3 scalefex_main.py
   ```

   Alternatively, you can specify the parameter file location calling the code this way:
   ```
   python scalefex_main.py -p parameters_test.yaml
   ```
   If you want to deploy ScaleFExSM on a notebook, look at the example described in the **Example** section

   
   If running the code on AWS:
   Deploy the 'ScaleFEx_main.yaml' Cloudformation template available [here](https://github.com/NYSCF/ScaleFEx/blob/main/Templates/ScaleFEx_main.yaml) and set your parameters.
   A detailed guide is available [here](https://github.com/NYSCF/ScaleFEx/wiki/ScaleFEx_on_AWS)

   
### Example

To see the output of ScaleFEx, execute the Jupyter Notebook provided with the code: **test_sample_dataset.ipynb**

If you leave the parameters.yaml file as it is, the notebook will automatically compute scalefex on the provided data. The notebook already displays that same output.



### License

ScaleFEx℠ is released under the BSD-3-Clause Clear license. For more details read the [LICENSE file](https://github.com/NYSCF/ScaleFEx/blob/main/LICENSE.md).

