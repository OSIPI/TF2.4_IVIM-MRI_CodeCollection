%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Function to calculate the IVIM-DTI signal from a measurement with many
% b-bvalues and diffusion directions.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input values: 
%   - params:               model parameters
%                           params(1) = S0 signal
%                           params(2:7) = diffusion tensor, given as 
%                           n x 6 vector. The order of the tensor elemts
%                           is: Dxx, Dyy, Dzz, Dxy, Dxz, Dyz
%                           params(8) = perfusion fraction f
%                           params(9) = pseudo-diffusion coefficient D*                           
%
%   - diffparams:           structure containing the fields 'bval' with the
%                           diffusion b-values and 'diffdir', containing the 
%                           gradient directinos. 
%                           In case of two-step fitting, the diffusion
%                           tensor is fixed and diffparams contains an
%                           additional field 'tensor'. 
% 
% The b-matrix is calculated from the b-values and gradient directions. 
% The signal is calculated according to 
% S = S0*(f*exp(-b*Ds) + (1-f)*exp(-bmat*diff_tensor))
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function signal = ivimdtifun(params, diffparams)

%check input parameters
if ndims(params) ~= 2 %#ok<ISMAT>
    error('Wrong matrix size of input parameters')
end

if isempty(find(size(params)==9,1)) && isempty(find(size(params)==3,1))
    error('Wrong number of input parameters')
end

if ~isstruct(diffparams)
    error('Diffparams must contain b-values and diffusion gradient directions')
end
if (~isfield(diffparams, 'bval') || ~isfield(diffparams, 'diffdir'))
    error('Diffparams must contain b-values and diffusion gradient directions')
end
if isempty(find(size(params)==9,1)) && ~isfield(diffparams, 'tensor')
    error('For two-step fitting diffparams must contain the diffusion tensor.')
end

%calculate b-matrix
bval = bval_scaling(diffparams.bval);
diffdir = diffparams.diffdir;

bmat = calc_bmat(bval, diffdir);

%calculate signal
if min(size(params))==1
    %single voxel
    S0 = params(1);
    if length(params) ~= 9
        %tensor fixed
        tensor = diffparams.tensor;
    else
        %tensor fitted
        tensor = params(2:7);
    end

    f = params(end-1);
    Ds = params(end);
    
else
    %multiple voxels
    if isempty(find(size(params)==9,1))
        %tensor fixed
        if size(params,1) ~=3
            params = params.';
        end
        tensor = diffparams.tensor;
    else
        %tensor fitted
        if size(params,1) ~=9
            params = params.';
        end
        tensor = params(2:7,:);
    end

    S0 = params(1,:);
    
    f = params(end-1,:);
    Ds = params(end,:);
end

signal = S0*(f*exp(-bval*Ds) + (1-f)*exp(-bmat*tensor.'));

