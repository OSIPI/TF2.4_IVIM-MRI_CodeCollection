function batchResultsFolder = run_QAMPER_IVIM(img_nii,bval_file,roi_nii,save_folder,optionS)

% function batchResultsFolder = run_QAMPER_IVIM(img_nii,bval_file,roi_nii,save_folder,optionS)
%   Inputs: image_nii   (char): path to 4-D NIfTI DWI volume with all b-values
%           bval_file   (char): path to .bval text file with b-value for each volume
%           roi_nii     (char): path to NIfTI ROI with segmentation mask for analysis region
%           save_folder (char): path to parent folder for results. Routine will create a subfolder under this location named 'dwi_maps'
%           optionS   (struct): additional processing options for QAMPER
%               'roiType' (char, default = 'node')
%               'avgRepeatBvals' (int, default = 1 (True))
%               'channels' (int, default = 16) the number of receiver channels/elements present in the coil (depending on organ of choice) for the DW-MRI data acquisition. 
%                   This factor is considered in noise correction calculation, please consult the details for MRI coil procured from the MRI Vendor to get this number.
%               'N_avgs' (int, default = 2) the number averages (DICOM TAG: 0018,0083) used for DW-MRI data acquisition (Navg = 2 or 4, etc., for each b-value). 
%
%   Dependencies: nii_toolbox (https://matlab.mathworks.com/open/fileexchange/v1?id=8797), included
%
%

if nargin < 5
    batchStruct = [];
else
    batchStruct = optionS;
end

batchStruct.batch_id = num2str(now);

DWIstruct.runDWI = 1;
% [~,fbase,ext] = fileparts(img_nii);

batchStruct.currentNiiPath = fullfile(save_folder,'dwi_maps');
batchResultsFolder = fullfile(save_folder,'dwi_maps',batchStruct.batch_id);

% if strcmp(ext,'.gz')
%     [~,fbase,~] = fileparts(fbase);
% end
% 
% bval_mat = fullfile(droot,[fbase '.mat']);
bval_arr = load(bval_file);
% 

DWIstruct.bval_arr  = bval_arr;

img = nii_load(img_nii,1);
vol_num = size(img.img,4);

DWIstruct.files{1} = img_nii; DWIstruct.vols = vol_num;
DWIstruct.files{2} = roi_nii;


DWIstruct.ADC = 0;
DWIstruct.NGIVIM=0;
DWIstruct.Kurtosis = 0;

DWIstruct.IVIM = 1;

if ~isfield(DWIstruct,'roiType')
    DWIstruct.roiType = 'node';
end

if ~isfield(DWIstruct,'avgRepeatBvals')
    DWIstruct.avgRepeatBvals = 1;
end

if ~isfield(DWIstruct,'channels')
    DWIstruct.channels = 16; %
end

if ~isfield(DWIstruct, 'N_avgs')
    DWIstruct.N_avgs = 2; %
end

if ~isfield(DWIstruct,'LB_IVIM')
    IVIMBoundsTable = [{0} {0} {0} {0}; {0.5} {0.005} {0.5} {1.0}; {0.05} {0.0005} {0.01} {0.5}];
    
    DWIstruct.LB_IVIM = cell2mat(IVIMBoundsTable(1,:));
    DWIstruct.UB_IVIM = cell2mat(IVIMBoundsTable(2,:));
    DWIstruct.x0_IVIM = cell2mat(IVIMBoundsTable(3,:));
end
% end

if ~isfield(DWIstruct,'Parallel')
    DWIstruct.Parallel = 1;
end

DWIstruct.bval_tf = ones(size(DWIstruct.bval_arr));
DWIstruct.nanopt = 1;

if ~isfield(DWIstruct,'kernelsize')
    DWIstruct.kernelsize = 1;
end
if ~isfield(DWIstruct,'smoothopt')
    DWIstruct.smoothopt = 1;
end

DWIstruct.previewMode = 0;
DWIstruct.previewAxes = 0;

batchStruct.DWIstruct = DWIstruct;
batchStruct.T1struct.runT1 = 0;
batchStruct.DCEstruct.runDCE = 0;
batchStruct.T2struct.runT2 = 0;

% save the batchfile for the job in folder with input image and in
% output folder
save(fullfile(save_folder,['batchStruct_' batchStruct.batch_id '.mat']),'batchStruct');

% run the batch
runQAMPERBatch(batchStruct);
disp('Batch complete.');