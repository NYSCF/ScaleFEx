# NYSCF_embeddings_extraction
Core function to compute the embeddings on our internal datasets. The scope is to refactor our internal pipelines to be more solid and easier to be updated from different components of the group

├── batch_compute_embeddings.py <br>
├── data_query<br>
│   ├── __init__.py<br>
│   ├── __pycache__<br>
│   │   ├── __init__.cpython-310.pyc<br>
│   │   ├── __init__.cpython-311.pyc<br>
│   │   ├── query_functions_AWS.cpython-311.pyc<br>
│   │   ├── query_functions_local.cpython-310.pyc<br>
│   │   ├── query_functions_local.cpython-311.pyc<br>
│   │   └── query_functions_local.cpython-39.pyc<br>
│   ├── query_functions_AWS.py<br>
│   ├── query_functions_local.py<br>
│   ├── README.md<br>
│   ├── requirements.txt<br>
│   ├── sample_data<br>
│   │   ├── ffc<br>
│   │   │   └── ffc.npy<br>
│   │   ├── rxrx2<br>
│   │   │   ├── images<br>
│   │   │   │   └── HUVEC-1<br>
│   │   │   │       ├── Plate1<br>
│   │   │   │       └── Plate2<br>
│   │   │   └── test_query_data_expected_output.csv<br>
│   │   └── zstack<br>
│   ├── test_query_functions_local.py<br>
│   └── try_functions.ipynb<br>
├── embeddings_extraction_class.py<br>
├── Load_preprocess_images<br>
│   ├── image_preprocessing_functions.py<br>
│   ├── __pycache__<br>
│   │   └── image_preprocessing_functions.cpython-311.pyc<br>
│   └── README.md<br>
├── Nuclei_segmentation<br>
│   ├── __init__.py<br>
│   ├── nuclei_location_extraction.py<br>
│   ├── __pycache__<br>
│   │   ├── __init__.cpython-311.pyc<br>
│   │   └── nuclei_location_extraction.cpython-311.pyc<br>
│   └── README.md<br>
├── parallelize_computation.py<br>
├── parameters.yaml<br>
├── __pycache__<br>
│   ├── batch_compute_embeddings.cpython-311.pyc<br>
│   ├── embeddings_extraction_class.cpython-311.pyc<br>
│   ├── parallelize_computation.cpython-310.pyc<br>
│   └── parallelize_computation.cpython-311.pyc<br>
├── Quality_control_HCI<br>
│   ├── compute_global_values.py<br>
│   ├── __pycache__<br>
│   │   └── compute_global_values.cpython-311.pyc<br>
│   └── README.md<br>
├── README.md<br>
├── test.ipynb<br>
└── try_functions.ipynb<br>
