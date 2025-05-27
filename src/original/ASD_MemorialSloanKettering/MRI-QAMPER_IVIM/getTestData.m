function [img_nii,bval_file,roi_nii] = getTestData

[d,~,~] = fileparts(which('getTestData'));
test_data_folder = fullfile(d,'test_data');
if ~exist(test_data_folder,'dir')
    mkdir(test_data_folder);
end

zenodo_url = 'https://zenodo.org/records/14605039/files/OSIPI_TF24_data_phantoms.zip';
save_zip = fullfile(test_data_folder,'OSIPI_TF24_data_phantoms.zip');
websave(save_zip,zenodo_url);
unzip(save_zip, test_data_folder);

img_nii = fulllfile(test_data_folder,'Data','brain.nii.gz');
bval_file = fulllfile(test_data_folder,'Data','brain.bval');
roi_nii = fulllfile(test_data_folder,'Data','brain_mask_gray_matter.nii.gz');
