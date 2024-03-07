# NYSCF_embeddings_extraction
Core function to compute the embeddings on our internal datasets. The scope is to refactor our internal pipelines to be more solid and easier to be updated from different components of the group
├── batch_compute_embeddings.py
├── data_query
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-310.pyc
│   │   ├── __init__.cpython-311.pyc
│   │   ├── query_functions_AWS.cpython-311.pyc
│   │   ├── query_functions_local.cpython-310.pyc
│   │   ├── query_functions_local.cpython-311.pyc
│   │   └── query_functions_local.cpython-39.pyc
│   ├── query_functions_AWS.py
│   ├── query_functions_local.py
│   ├── README.md
│   ├── requirements.txt
│   ├── sample_data
│   │   ├── ffc
│   │   │   └── ffc.npy
│   │   ├── rxrx2
│   │   │   ├── images
│   │   │   │   └── HUVEC-1
│   │   │   │       ├── Plate1
│   │   │   │       │   ├── AB08_s1_w1.png
│   │   │   │       │   ├── AB08_s2_w1.png
│   │   │   │       │   ├── AB08_s3_w1.png
│   │   │   │       │   ├── AB08_s4_w1.png
│   │   │   │       │   ├── L18_s1_w1.png
│   │   │   │       │   ├── L18_s2_w1.png
│   │   │   │       │   ├── L18_s3_w1.png
│   │   │   │       │   └── L18_s4_w1.png
│   │   │   │       └── Plate2
│   │   │   │           ├── AB08_s1_w1.png
│   │   │   │           ├── AB08_s2_w1.png
│   │   │   │           ├── AB08_s3_w1.png
│   │   │   │           ├── AB08_s4_w1.png
│   │   │   │           ├── L18_s1_w1.png
│   │   │   │           ├── L18_s2_w1.png
│   │   │   │           ├── L18_s3_w1.png
│   │   │   │           └── L18_s4_w1.png
│   │   │   └── test_query_data_expected_output.csv
│   │   └── zstack
│   │       ├── max_proj.tiff
│   │       ├── r11c06f23p01-ch1sk1fk1fl1.tiff
│   │       ├── r11c06f23p02-ch1sk1fk1fl1.tiff
│   │       ├── r11c06f23p03-ch1sk1fk1fl1.tiff
│   │       └── r11c06f23p04-ch1sk1fk1fl1.tiff
│   ├── test_query_functions_local.py
│   └── try_functions.ipynb
├── embeddings_extraction_class.py
├── Load_preprocess_images
│   ├── image_preprocessing_functions.py
│   ├── __pycache__
│   │   └── image_preprocessing_functions.cpython-311.pyc
│   └── README.md
├── Nuclei_segmentation
│   ├── __init__.py
│   ├── nuclei_location_extraction.py
│   ├── __pycache__
│   │   ├── __init__.cpython-311.pyc
│   │   └── nuclei_location_extraction.cpython-311.pyc
│   └── README.md
├── parallelize_computation.py
├── parameters.yaml
├── __pycache__
│   ├── batch_compute_embeddings.cpython-311.pyc
│   ├── embeddings_extraction_class.cpython-311.pyc
│   ├── parallelize_computation.cpython-310.pyc
│   └── parallelize_computation.cpython-311.pyc
├── Quality_control_HCI
│   ├── compute_global_values.py
│   ├── __pycache__
│   │   └── compute_global_values.cpython-311.pyc
│   └── README.md
├── README.md
├── test.ipynb
└── try_functions.ipynb
