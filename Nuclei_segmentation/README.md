[![testing status](https://github.com/NYSCF/Nuclei_segmentation/actions/workflows/python-package.yml/badge.svg)](https://github.com/NYSCF/Nuclei_segmentation/actions/workflows/python-package.yml)
[![python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
# Nuclei_segmentation
Collection of methods for nucleus segmentation used as a submodule for the ScaleFEx pipeline

Functions:
- `compute_DNA_mask()`: compute binary masks using triangle thresholding and some morphological operations for smoothing/cleaning
- `retrieve_coordinates()`: retrieves coordinates of centers of cells from masks computed via `compute_DNA_mask()` that have an area within in a specified range

Other submodules used for the same pipeline include:
  - [`NYSCF/ScalefFEx_from_crop`](https://github.com/NYSCF/ScaleFEx_from_crop)
  - [`NYSCF/data_query`](https://github.com/NYSCF/data_query)
  - [`NYSCF/Quality_control_HCI`](https://github.com/NYSCF/Quality_control_HCI)