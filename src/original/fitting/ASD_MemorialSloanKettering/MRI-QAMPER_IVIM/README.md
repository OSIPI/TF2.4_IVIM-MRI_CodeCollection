# MRI-QAMPER_IVIM

This is the codebase for intravoxel incoherent motion (IVIM) code from the MRI-QAMPER MATLAB package, developed by Dr. Shukla-Dave's lab at Memorial Sloan Kettering Cancer Center.

__Authors:__ Eve LoCastro (locastre@mskcc.org), Dr. Ramesh Paudyal (paudyalr@mskcc.org), Dr. Amita Shukla-Dave (davea@mskcc.org) </br>
__Institution:__ Memorial Sloan Kettering Cancer Center </br>
__Department:__ Medical Physics</br>
__Address:__ 321 E 61st St, New York, NY 10022

The codebase and subdirectories should be added to the MATLAB path. An example usage script is provided in `demo_QAMPER_IVIM.m`.

```
% This is an example usage script for MSK Medical Physics Dave Lab QAMPER IVIM
% Please replace the variable names below with path values for 
%       qamper_path: path to MRI-QAMPER_IVIM folder
%       img_nii: multi-b value DWI (4-D NIfTI)
%       bval_file: b-value information (txt file)
%       roi_nii: single-volume mask ROI image (NIfTI)

qamper_path = 'path:\to\MRI-QAMPER_IVIM';
addpath(genpath(qamper_path));

img_nii = 'dwi.nii';
bval_file = 'dwi.bval';
roi_nii = 'roi.nii';

save_folder = fullfile(qamper_path,'test_data');

batchResultsFolder = run_QAMPER_IVIM(img_nii,bval_file,roi_nii,save_folder);
```