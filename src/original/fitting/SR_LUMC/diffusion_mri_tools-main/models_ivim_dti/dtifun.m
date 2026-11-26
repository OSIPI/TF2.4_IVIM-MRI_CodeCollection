%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Function to calculate the signal from a DTI measurement given the
% diffusion tensor. 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input values: 
%   - diffusion tensor:     given as n x 7 vector
%                           The first entry of the diffusion vector is S0. 
%                           Order of the tensor elements tensor(2:7): 
%                           Dxx, Dyy, Dzz, Dxy, Dxz, Dyz
%
%   - b-matrix:             given as (n x 6 vector)
% 
% The signal is calculated according to S = S0*exp(-bmat*diff_tensor)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function signal = dtifun(tensor, bmat)

if ndims(tensor) ~= 2 %#ok<ISMAT>
    error('wrong matrix size')
end
if isempty(find(size(tensor)==7, 1))
    error('no valid diffusion tensor provided')
end

%check sizes and calculate signal
if min(size(tensor))==1
    %single voxel
    signal = tensor(1)*exp(-bmat*tensor(2:7).');
else
    if size(tensor,1) ~= 7
        tensor = tensor.';
    end
    %multiple voxels
    signal = tensor(1,:).*exp(-bmat*tensor(2:7,:));
end

