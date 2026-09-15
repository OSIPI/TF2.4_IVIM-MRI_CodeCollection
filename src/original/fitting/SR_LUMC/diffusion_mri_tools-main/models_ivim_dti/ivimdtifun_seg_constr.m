%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Function to calculate the IVIM-DTI signal given the lower triangular
% matrix (from the Cholesky decomposition) of the diffusion tensor and the
% perfusion fraction f. The segmented IVIM-DTI equation is used.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input values: 
%   - params:               model parameters
%                           params(1) = S0 signal
%                           params(2:7) = lower triangular matrix of the 
%                           diffusion tensor
%                           params(8) = perfusion fraction f
%
%   - diffparams:           structure containing the fields 'bval' with the
%                           diffusion b-values and 'diffdir', containing the 
%                           gradient directinos. 
% 
% First, the diffusion tensor is calculated from the lower triangular
% matrix using the function lower_triangular2tensor. 
% Then, the signal is calculated using the function ivimdtifun_seg:
% S = S0*(f*delta_fun(bval) + (1-f)*exp(-bmat*diff_tensor))
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function signal = ivimdtifun_seg_constr(params, diffparams)

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

%Calculate tensor from Cholesky decomposition
if min(size(params)) == 1
    %single voxel
    params(2:7) = lower_triangular2tensor(params(2:7));
else
    if size(params,1) ~= 8
        params = params.';
    end
    params(2:7,:) = lower_triangular2tensor(params(2:7,:));
end

%Calculate signal
signal = ivimdtifun_seg(params, diffparams);



