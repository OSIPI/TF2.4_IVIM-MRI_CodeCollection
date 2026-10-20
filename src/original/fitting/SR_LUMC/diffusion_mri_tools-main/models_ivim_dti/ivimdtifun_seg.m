%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Function to calculate the IVIM-DTI signal from a measurement with many
% b-bvalues and diffusion directions using the segmented IVIM-DTI equation.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input values: 
%   - params:               model parameters
%                           params(1) = S0 signal
%                           params(2:7) = diffusion tensor, given as 
%                           n x 7 vector. The order of the tensor elemts
%                           is: Dxx, Dyy, Dzz, Dxy, Dxz, Dyz
%                           params(8) = perfusion fraction f
%
%   - diffparams:           structure containing the fields 'bval' with the
%                           diffusion b-values and 'diffdir', containing the 
%                           gradient directinos. 
% 
% The b-matrix is calculated from the b-values and gradient directions. 
% The signal is calculated according to 
% S = S0*(f*delta_fun(bval) + (1-f)*exp(-bmat*diff_tensor))
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function signal = ivimdtifun_seg(params, diffparams)

%check input parameters
if ndims(params) ~= 2 %#ok<ISMAT>
    error('Wrong matrix size of input parameters')
end

if length(params)>8
    warning('Too many input parameters provided, additional parameters are ignored')
end

if ~isstruct(diffparams)
    error('Diffparams must be a structure containing b-values and diffusion gradient directions')
end
if (~isfield(diffparams, 'bval') || ~isfield(diffparams, 'diffdir'))
    error('Diffparams must contain b-values and diffusion gradient directions')
end

%calculate b-matrix
bval = bval_scaling(diffparams.bval);
diffdir = diffparams.diffdir;

bmat = calc_bmat(bval, diffdir);

%calculate signal
if min(size(params))==1
    %single voxel
    S0 = params(1);
    tensor = params(2:7);
    f = params(8);
else
    %multiple voxels
    if size(params,1) ~= 8
        params = params.';
    end
    tensor = params(2:7,:);
    S0 = params(1,:);    
    f = params(8,:);
end

signal = S0*(f*delta_fun(bval) + (1-f)*exp(-bmat*tensor.'));

