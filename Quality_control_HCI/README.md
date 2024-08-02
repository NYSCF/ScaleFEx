# Quality_control_HCI
Performs a first pass of whole-image quality control for high content imaging screens

## Recorded Statistics

- `tot_cell_num`: total number of cells in image (live + dead)    
- `Cell_num`: number of live cells in image
- `Max_Intensity`: max pixel intensity across the entire image
- `Min_Intensity`: min pixel intensity across the entire image
- `Mean_Intensity`: mean pixel intensity across the entire image
- `Blur`: variance of inverse of laplacian scaled by cell count (see [this link](https://pyimagesearch.com/2015/09/07/blur-detection-with-opencv/))
- `InFocus`: boolean based on preset `Blur` threshold (2700). this is very dataset dependent, so your mileage may vary when using our threshold for a different dataset
- `Usable`: False if InFocus is False, and vice versa
- `Mean_Foreground_Intensity`: mean pixel intensity of foreground identified using Otsu's method
- `Mean_Background_Intensity`: mean pixel intensity of background identified using Otsu's method
- `SNR`: signal-to-noise ratio
- `neural_len`: sum of lengths of all neurites (if any)