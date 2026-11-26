% Set-up: Add all subfolders to the path
% Run setup.m once before using any project functions. 

diffusion_mri_tools = fileparts(mfilename('fullpath'));
addpath(genpath(diffusion_mri_tools));