# Nuclei_segmentation
Collection of methods for nucleus segmentation used as a submodule for the ScaleFEx pipeline

Functions:
- `compute_DNA_mask()`: compute binary masks using triangle thresholding and some morphological operations for smoothing/cleaning
- `retrieve_coordinates()`: retrieves coordinates of centers of cells from masks computed via `compute_DNA_mask()` that have an area within in a specified range

Other submodules used for the same pipeline include:
  - [`NYSCF/ScalefFEx_from_crop`](https://github.com/NYSCF/ScaleFEx_from_crop)
  - [`NYSCF/data_query`](https://github.com/NYSCF/data_query)
  - [`NYSCF/Quality_control_HCI`](https://github.com/NYSCF/Quality_control_HCI)