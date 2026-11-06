%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Function to calculate the IVIM-DTI signal given the lower triangular
% matrix (from the Cholesky decomposition) of the diffusion tensor,
% perfusion fraction f and pseudo-diffusion coefficient D_star.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input values:
%   - params:               model parameters
%                           params(1) = S0 signal
%                           params(2:7) = lower triangular matrix of the
%                           diffusion tensor
%                           params(8) = perfusion fraction f
%                           params(9) = pseudo-diffusion coefficient D*
%
%   - diffparams:           structure containing the fields 'bval' with the
%                           diffusion b-values and 'diffdir', containing the
%                           gradient directinos.
%
% First, the diffusion tensor is calculated from the lower triangular
% matrix using the function lower_triangular2tensor.
% Then, the signal is calculated using the function ivimdtifun:
% S = S0*(f*exp(-b*Ds) + (1-f)*exp(-bmat*diff_tensor))
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function signal = ivimdtifun_constr(params, diffparams)

if ndims(params) ~= 2 %#ok<ISMAT>
    error('Wrong matrix size of input parameters')
end
if isempty(find(size(params)==9,1))
    error('Wrong number of input parameters')
end

%Calculate tensor from Cholesky decomposition
if min(size(params))==1
    %single voxel
    params(2:7) = lower_triangular2tensor(params(2:7));
else
    if size(params,1) ~= 9
        params = params.';
    end
    params(2:7,:) = lower_triangular2tensor(params(2:7,:));
end

%Calculate signal
signal = ivimdtifun(params, diffparams);

