[![testing status](https://github.com/NYSCF/data_query/actions/workflows/python-package.yml/badge.svg)](https://github.com/NYSCF/data_query/actions/workflows/python-package.yml)
[![python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
# data_query
Module for data query, checking for file existence, and generating dataframe with images to process. This repo is designed to be generalizable to different datasets.

## Data Structure
The structure of your data should generally adhere to the following structure:
```
└── experiment
    ├── plate1
    │   ├── well1_site1_ch1.png
    │   ├── well1_site2_ch1.png
    │   ├── well2_site1_ch1.png
    │   └── well2_site2_ch1.png
    └── plate2
        ├── well1_site1_ch1.png
        ├── well1_site2_ch1.png
        ├── well2_site1_ch1.png
        └── well2_site2_ch1.png
```
with a main "experiment" directory containing a folder for each plate's images you wish to analyze. 

There can be any number of subfolders between a plate folder (e.g. `experiment/plate1/`) and its images, as long as each plate's images follows the same structure with the same subfolder names.

A **correct** example:
```
└── experiment
    ├── plate1
    │   └── images
    │       ├── well1_site1_ch1.png
    │       ├── well1_site2_ch1.png
    │       ├── well2_site1_ch1.png
    │       └── well2_site2_ch1.png
    └── plate2
        └── images
            ├── well1_site1_ch1.png
            ├── well1_site2_ch1.png
            ├── well2_site1_ch1.png
            └── well2_site2_ch1.png
```
An **incorrect** example:
```
└── experiment
    ├── plate1
    │   └── images
    │       ├── well1_site1_ch1.png
    │       ├── well1_site2_ch1.png
    │       ├── well2_site1_ch1.png
    │       └── well2_site2_ch1.png
    └── plate2
        └── imgs <--- different name
            ├── well1_site1_ch1.png
            ├── well1_site2_ch1.png
            ├── well2_site1_ch1.png
            └── well2_site2_ch1.png
```

## Using `query_data()`
Parameters:
1. `exp_folder`: main directory with a subdirectory for each plate you would like to get images for
2. `pattern`: arrangement of info in filenames to parse, such as `well`, `site`, `channel`, etc. (Explained in detail below)
3. `plates`: ids of desired plates (each must have `plate_identifiers` that come immediately before and after)
4. `plate_identifiers`: two substrings that immediately precede and succeed each plate in `plates` 
5. `exts`: file extensions of your images (without a period) (e.g., `'png'`,`'tiff'`,`'jpg'`)

### Using `pattern` for your data:
Say you have data structured like this:
```
└── HUVEC-1 
    ├── Plate1
    │   ├── AB08_s1_w1.png
    │   ├── AB08_s2_w1.png
    │   ├── AB08_s3_w1.png
    │   ├── AB08_s4_w1.png
    │   ├── L18_s1_w1.png
    │   ├── L18_s2_w1.png
    │   ├── L18_s3_w1.png
    │   └── L18_s4_w1.png
    └── Plate2 
        ├── AB08_s1_w1.png
        ├── AB08_s2_w1.png
        ├── AB08_s3_w1.png
        ├── AB08_s4_w1.png
        ├── L18_s1_w1.png
        ├── L18_s2_w1.png
        ├── L18_s3_w1.png
        └── L18_s4_w1.png
```
as is the case for the [RxRx2](https://www.rxrx.ai/rxrx2) test images in the `tests/sample_data/rxrx2` directory of this repo.

Consider the image path: 
```
'/path/to/HUVEC-1/Plate1/AB08_s1_w1.png'
```

The `exp_folder` parameter in this case would be `'/path/to/HUVEC-1/'` and `plate_identifiers` could be something like `['Plate','']`.

The `pattern` parameter is meant to help with parsing relevant metadata encoded in filenames of imaging datasets, like:
- well
- site/field
- channel
- plane

for the data above, the filenames have a pattern of `'<well>_<site>_<channel>.<ext>` (where `<ext>` will substituted for each of the extensions list in the `exts` parameter)

Using this function to parse this sample data will look like this:
```
files_df = query_data_updated(exp_folder='/path/to/HUVEC-1/', plate_identifiers=['Plate',''],
                                pattern='<well>_<site>_<channel>.<ext>,exts=['png'])
```
where `files_df` will be a pandas DataFrame with this structure:
```
     plate  well site channel    plane        filename                               file_path
0   Plate1  AB08   s1      w1  plane01  AB08_s1_w1.png  /path/to/HUVEC-1/Plate1/AB08_s1_w1.png
1   Plate1  AB08   s2      w1  plane01  AB08_s2_w1.png  /path/to/HUVEC-1/Plate1/AB08_s2_w1.png
2   Plate1  AB08   s3      w1  plane01  AB08_s3_w1.png  /path/to/HUVEC-1/Plate1/AB08_s3_w1.png
3   Plate1  AB08   s4      w1  plane01  AB08_s4_w1.png  /path/to/HUVEC-1/Plate1/AB08_s4_w1.png
4   Plate1   L18   s1      w1  plane01   L18_s1_w1.png   /path/to/HUVEC-1/Plate1/L18_s1_w1.png
5   Plate1   L18   s2      w1  plane01   L18_s2_w1.png   /path/to/HUVEC-1/Plate1/L18_s2_w1.png
6   Plate1   L18   s3      w1  plane01   L18_s3_w1.png   /path/to/HUVEC-1/Plate1/L18_s3_w1.png
7   Plate1   L18   s4      w1  plane01   L18_s4_w1.png   /path/to/HUVEC-1/Plate1/L18_s4_w1.png
8   Plate2  AB08   s1      w1  plane01  AB08_s1_w1.png  /path/to/HUVEC-1/Plate2/AB08_s1_w1.png
9   Plate2  AB08   s2      w1  plane01  AB08_s2_w1.png  /path/to/HUVEC-1/Plate2/AB08_s2_w1.png
10  Plate2  AB08   s3      w1  plane01  AB08_s3_w1.png  /path/to/HUVEC-1/Plate2/AB08_s3_w1.png
11  Plate2  AB08   s4      w1  plane01  AB08_s4_w1.png  /path/to/HUVEC-1/Plate2/AB08_s4_w1.png
12  Plate2   L18   s1      w1  plane01   L18_s1_w1.png   /path/to/HUVEC-1/Plate2/L18_s1_w1.png
13  Plate2   L18   s2      w1  plane01   L18_s2_w1.png   /path/to/HUVEC-1/Plate2/L18_s2_w1.png
14  Plate2   L18   s3      w1  plane01   L18_s3_w1.png   /path/to/HUVEC-1/Plate2/L18_s3_w1.png
15  Plate2   L18   s4      w1  plane01   L18_s4_w1.png   /path/to/HUVEC-1/Plate2/L18_s4_w1.png
```
### Parameter Rules

1. Do **not** include any leading or following forward slashes (`/`) in the `pattern`,`plates`, or `plate_identifiers` parameters
2. You **must** include field character lengths in a `pattern` if consecutive metadata fields are not separated by any characters (e.g. `<well(6)><site(3)><plane(3)>-<channel(3)>.<ext>` is the pattern for Opera Phenix images `r11c06f23p02-ch1sk1fk1fl1.tiff` )
3. A `plate_identifier` of `''` indicates the beginning or end of plate directory name

#### Specifiying field lengths in a pattern:
In cases where you have consecutive fields that you would like separate, you can specify the number of characters in the `pattern` string. 

Images from the [Opera Phenix](https://www.revvity.com/product/operetta-cls-system-hh16000020) line of high-content imaging systems typically have filenames that follow this format:

`r11c06f23p02-ch1sk1fk1fl1.tiff`

and the corresponding `pattern` to parse these filenames would be:

`<well(6)><site(3)><plane(3)>-<channel(3)>.<ext>`

where `<well(6)><site(3)>` indicates that the well is 6 characters long, the site is 3 characters long after the well, and so on.

## Examples:

### [RxRx2](https://www.rxrx.ai/rxrx2):
Varying the `plate_identifiers` and `plates` parameters:
```
files_df = query_functions_local.query_data('tests/sample_data/rxrx2/images/HUVEC-1',
                                         pattern='<well>_<site>_<channel>.<ext>',
                                         plate_identifiers=['',''],
                                         plates=['Plate1'],
                                         exts=['png'])
```
gives a DataFrame with the structure
```
    plate  well site channel    plane        filename plate_folder                                                     file_path
0  Plate1  AB08   s1      w1  plane01  AB08_s1_w1.png       Plate1  tests/sample_data/rxrx2/images/HUVEC-1/Plate1/AB08_s1_w1.png
1  Plate1  AB08   s2      w1  plane01  AB08_s2_w1.png       Plate1  tests/sample_data/rxrx2/images/HUVEC-1/Plate1/AB08_s2_w1.png
2  Plate1  AB08   s3      w1  plane01  AB08_s3_w1.png       Plate1  tests/sample_data/rxrx2/images/HUVEC-1/Plate1/AB08_s3_w1.png
3  Plate1  AB08   s4      w1  plane01  AB08_s4_w1.png       Plate1  tests/sample_data/rxrx2/images/HUVEC-1/Plate1/AB08_s4_w1.png
4  Plate1   L18   s1      w1  plane01   L18_s1_w1.png       Plate1   tests/sample_data/rxrx2/images/HUVEC-1/Plate1/L18_s1_w1.png
5  Plate1   L18   s2      w1  plane01   L18_s2_w1.png       Plate1   tests/sample_data/rxrx2/images/HUVEC-1/Plate1/L18_s2_w1.png
6  Plate1   L18   s3      w1  plane01   L18_s3_w1.png       Plate1   tests/sample_data/rxrx2/images/HUVEC-1/Plate1/L18_s3_w1.png
7  Plate1   L18   s4      w1  plane01   L18_s4_w1.png       Plate1   tests/sample_data/rxrx2/images/HUVEC-1/Plate1/L18_s4_w1.png
```
Setting `plates=[1]` and `plate_identifiers=['Plate','']` results in:
```
  plate  well site channel    plane        filename plate_folder                                                     file_path
0     1  AB08   s1      w1  plane01  AB08_s1_w1.png       Plate1  tests/sample_data/rxrx2/images/HUVEC-1/Plate1/AB08_s1_w1.png
1     1  AB08   s2      w1  plane01  AB08_s2_w1.png       Plate1  tests/sample_data/rxrx2/images/HUVEC-1/Plate1/AB08_s2_w1.png
2     1  AB08   s3      w1  plane01  AB08_s3_w1.png       Plate1  tests/sample_data/rxrx2/images/HUVEC-1/Plate1/AB08_s3_w1.png
3     1  AB08   s4      w1  plane01  AB08_s4_w1.png       Plate1  tests/sample_data/rxrx2/images/HUVEC-1/Plate1/AB08_s4_w1.png
4     1   L18   s1      w1  plane01   L18_s1_w1.png       Plate1   tests/sample_data/rxrx2/images/HUVEC-1/Plate1/L18_s1_w1.png
5     1   L18   s2      w1  plane01   L18_s2_w1.png       Plate1   tests/sample_data/rxrx2/images/HUVEC-1/Plate1/L18_s2_w1.png
6     1   L18   s3      w1  plane01   L18_s3_w1.png       Plate1   tests/sample_data/rxrx2/images/HUVEC-1/Plate1/L18_s3_w1.png
7     1   L18   s4      w1  plane01   L18_s4_w1.png       Plate1   tests/sample_data/rxrx2/images/HUVEC-1/Plate1/L18_s4_w1.png
```

### Opera Phenix:
Here is an example Phenix image directory structure for an experiment called `'EXP0001'`:
```
├── EXP0001
│   ├── EXP0001_CCU384_401__2024-02-17T22_04_34-Measurement 19
│   │   └── Images
│   │       ├── r03c07f01p01-ch1sk1fk1fl1.tiff
│   │       ├── r03c07f01p01-ch2sk1fk1fl1.tiff
│   │       ├── r06c17f01p01-ch1sk1fk1fl1.tiff
│   │       └── r06c17f01p01-ch2sk1fk1fl1.tiff
│   ├── EXP0001_CCU384_402__2024-02-13T22_52_28-Measurement 19
│   │   └── Images
│   │       ├── r03c07f01p01-ch1sk1fk1fl1.tiff
│   │       ├── r03c07f01p01-ch2sk1fk1fl1.tiff
│   │       ├── r06c17f01p01-ch1sk1fk1fl1.tiff
│   │       └── r06c17f01p01-ch2sk1fk1fl1.tiff
│   ├── EXP0001_CV384W_201__2024-06-15T05_02_40-Measurement 1
│   │   └── Images
│   │       ├── r03c07f01p01-ch1sk1fk1fl1.tiff
│   │       ├── r03c07f01p01-ch2sk1fk1fl1.tiff
│   │       ├── r06c17f01p01-ch1sk1fk1fl1.tiff
│   │       └── r06c17f01p01-ch2sk1fk1fl1.tiff
│   └── EXP0001_CV384W_202__2024-06-15T05_09_49-Measurement 1
│       └── Images
│           ├── r03c07f01p01-ch1sk1fk1fl1.tiff
│           ├── r03c07f01p01-ch2sk1fk1fl1.tiff
│           ├── r06c17f01p01-ch1sk1fk1fl1.tiff
│           └── r06c17f01p01-ch2sk1fk1fl1.tiff
```
To get the metadata for the plates numbered `201` and `401`, we use these parameters:
```
files_df = query_functions_local.query_data('/home/jeffdatasci/test_imaging_data/EXP0001',
                                            pattern='Images/<Well(6)><Site(3)><Plane(3)>-<Channel(3)>.<ext>',
                                            plates=['201','401'],
                                            plate_identifiers=['_','_'],
                                            exts=['tiff'])
```
**NOTE:**
The metadata fields include character lengths (e.g., `<Well(6)>`) because metadata for well, site, etc are not separated by any other characters. Also, `pattern` includes the `Images` subdirectory of each plate directory. 


This function call results in a DataFrame of the structure:
```
  plate    well site channel plane                        filename  \
0   401  r03c07  f01     ch1   p01  r03c07f01p01-ch1sk1fk1fl1.tiff   
1   401  r03c07  f01     ch2   p01  r03c07f01p01-ch2sk1fk1fl1.tiff   
2   401  r06c17  f01     ch1   p01  r06c17f01p01-ch1sk1fk1fl1.tiff   
3   401  r06c17  f01     ch2   p01  r06c17f01p01-ch2sk1fk1fl1.tiff   
4   201  r03c07  f01     ch1   p01  r03c07f01p01-ch1sk1fk1fl1.tiff   
5   201  r03c07  f01     ch2   p01  r03c07f01p01-ch2sk1fk1fl1.tiff   
6   201  r06c17  f01     ch1   p01  r06c17f01p01-ch1sk1fk1fl1.tiff   
7   201  r06c17  f01     ch2   p01  r06c17f01p01-ch2sk1fk1fl1.tiff   

                                             plate_folder  \
0  EXP0001_CCU384_401__2024-02-17T22_04_34-Measurement 19   
1  EXP0001_CCU384_401__2024-02-17T22_04_34-Measurement 19   
2  EXP0001_CCU384_401__2024-02-17T22_04_34-Measurement 19   
3  EXP0001_CCU384_401__2024-02-17T22_04_34-Measurement 19   
4   EXP0001_CV384W_201__2024-06-15T05_02_40-Measurement 1   
5   EXP0001_CV384W_201__2024-06-15T05_02_40-Measurement 1   
6   EXP0001_CV384W_201__2024-06-15T05_02_40-Measurement 1   
7   EXP0001_CV384W_201__2024-06-15T05_02_40-Measurement 1   

                                                                                             file_path  
0  EXP0001/EXP0001_CCU384_401__2024-02-17T22_04_34-Measurement 19/Images/r03c07f01p01-ch1sk1fk1fl1....  
1  EXP0001/EXP0001_CCU384_401__2024-02-17T22_04_34-Measurement 19/Images/r03c07f01p01-ch2sk1fk1fl1....  
2  EXP0001/EXP0001_CCU384_401__2024-02-17T22_04_34-Measurement 19/Images/r06c17f01p01-ch1sk1fk1fl1....  
3  EXP0001/EXP0001_CCU384_401__2024-02-17T22_04_34-Measurement 19/Images/r06c17f01p01-ch2sk1fk1fl1....  
4  EXP0001/EXP0001_CV384W_201__2024-06-15T05_02_40-Measurement 1/Images/r03c07f01p01-ch1sk1fk1fl1.tiff  
5  EXP0001/EXP0001_CV384W_201__2024-06-15T05_02_40-Measurement 1/Images/r03c07f01p01-ch2sk1fk1fl1.tiff  
6  EXP0001/EXP0001_CV384W_201__2024-06-15T05_02_40-Measurement 1/Images/r06c17f01p01-ch1sk1fk1fl1.tiff  
7  EXP0001/EXP0001_CV384W_201__2024-06-15T05_02_40-Measurement 1/Images/r06c17f01p01-ch2sk1fk1fl1.tiff
```

## Questions:
If you have any issues/questions regarding this code, feel free to [open an issue](https://github.com/NYSCF/data_query/issues)




