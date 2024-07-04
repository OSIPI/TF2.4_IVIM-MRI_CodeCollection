function [img_nii,bval_file,roi_nii,save_folder] = getTestData

[d,~,~] = fileparts(which('getTestData'));

img_nii = fullfile(d,'test_data','702-HN401D-D2019_10_08.nii.gz');
bval_file = fullfile(d,'test_data','702-HN401D-D2019_10_08.bval');
roi_nii = fullfile(d,'test_data','702-HN401D-D2019_10_08_BD_REDO_SV.nii.gz');
save_folder = fullfile(d,'test_data');