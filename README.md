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
    - **vector_type**: Write 'scalefex' for the feature vector, 'embeddings' for the deep embeddings, '' if you want only the preprocessing part (specified below)
    - **resource**: 'local' foor local computation, 'AWS' for cloud computing
    - **n_of_workers**: 100 int, n of workers to use in parallel.
    - **exp_folder**: '/path/to/images/'
    - **experiment_name**: 'exp001' this name will be appended to the saved files
    - **saving_folder**: '/path/to/saving/folder/' path to the saving folder
    - **plates**: ['Plate1','Plate2'] if you want to process a subset of plates, 'all' for all of the ones found in the folder
    - **plate_type**: '_CV384W_' identifier for finding the plate number
    - **fname_pattern**: 'Images/<Well(6)><Site(3)><Plane(3)>-<Channel(3)>.<ext>' @Jeff can you describe?
    - **fname_delimiters**: ['-','.'] @Jeff can you describe?
    - **file_extensions**: ['tiff'] @Jeff can you describe?
    - **image_size**: [2160,2160] size of the image

# type_specific:
channel: ['ch4','ch1', 'ch5',  'ch3', 'ch2']        
stack: False
ROI: 150
neurite_tracing: ''
bf_channel: ''
RNA_channel: 'ch5' #set only if you want to compute ScaleFex
Mito_channel: 'ch2' #set only if you want to compute ScaleFex
zstack: False
compute_live_cells: False 

downsampling: 1
QC: True
FFC: False
FFC_n_images: 500
tile_computation: False

# segmentation:
csv_coordinates: '' # '' if you don't want to use a pre-computed coordinates file
segmenting_function: 'MaskRCNN_Deployment.segmentation_mrcnn' # 'Nuclei_segmentation.nuclei_location_extraction' # 'MaskRCNN_Deployment.segmentation_mrcnn'   
AI_cell_segmentation: True
save_coordinates: True
gpu_AI: False
Threshold_segmentation: True
use_cpu_segmentation: True
min_cell_size: 200 #area
max_cell_size: 100000
visualization: False

## embeddings_settings
device: "0"
weights_location: "/home/biancamigliori/Documents/"

## Scalefex settings
visualize_masks: False
visualize_crops: False

#AWS settings
s3_bucket: 'nyscf-scalefex'
subset: 'A'
3. **Execute Analysis**:
   ```
   python vector_extraction_class()
   ```
### Example

tbd

### Contributing

We welcome contributions to ScaleFEx℠. ? Please read `CONTRIBUTING.md` for guidance on submitting pull requests.

### License

ScaleFEx℠ is released under the MIT License. [To be updated] See [LICENSE.md](LICENSE.md) for more details.

### Acknowledgements
??? maybe
This project is supported by researchers dedicated to advancing drug discovery and understanding disease mechanisms through image-based cellular profiling.
