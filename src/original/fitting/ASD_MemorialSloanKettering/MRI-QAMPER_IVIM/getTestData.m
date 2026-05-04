function [img_nii,bval_file,roi_nii] = getTestData

[d,~,~] = fileparts(which('getTestData'));
d1 = fileparts(d);        % one level up
d2 = fileparts(d1);       % two levels up
d3 = fileparts(d2);       % three levels up
d4 = fileparts(d3);       % three levels up
test_data_folder = fullfile(d4, 'download');
if ~exist(test_data_folder,'dir')
    mkdir(test_data_folder);
end

if ~exist(fullfile(test_data_folder,'OSIPI_TF24_data_phantoms.zip'),'file')
    zenodo_url = 'https://zenodo.org/records/14605039/files/OSIPI_TF24_data_phantoms.zip';
    save_zip = fullfile(test_data_folder,'OSIPI_TF24_data_phantoms.zip');
    websave(save_zip,zenodo_url);
    unzip(save_zip, test_data_folder);
end

img_nii = fullfile(test_data_folder,'Data','brain.nii.gz');
bval_file = fullfile(test_data_folder,'Data','brain.bval');
roi_nii = fullfile(test_data_folder,'Data','brain_mask_gray_matter.nii.gz');
