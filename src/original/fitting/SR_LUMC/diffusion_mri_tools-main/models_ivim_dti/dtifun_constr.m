%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Function to calculate the diffusion signal given the lower triangular 
% matrix (from the Cholesky decomposition) of the diffusion tensor. 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input values: 
%   -   lower triangular matrix of the diffusion tensor 
%   -   b-matrix
% The diffusion tensor and b-matrix are given as n x 6 element vectors. The
% first entry of the diffusion tensor is the S0 signal. 
% 
% First, the diffusion tensor is calculated from the lower triangular
% matrix using the function lower_triangular2tensor. 
% Then, the signal is calculated according to S = S0*exp(-bmat*diff_tensor)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function signal = dtifun_constr(tensor, bmat)


if ndims(tensor) ~= 2 %#ok<ISMAT>
    error('wrong matrix size')
end
if isempty(find(size(tensor)==7, 1))
    error('no valid diffusion tensor provided')
end


%Calculate tensor from Cholesky decomposition
if min(size(tensor))==1
    %single voxel
    tensor(2:7) = lower_triangular2tensor(tensor(2:7));
else
    if size(tensor,1) ~= 7
        tensor = tensor.';
    end
    tensor(2:7,:) = lower_triangular2tensor(tensor(2:7,:));
end

%Calculate signal
signal = dtifun(tensor, bmat);