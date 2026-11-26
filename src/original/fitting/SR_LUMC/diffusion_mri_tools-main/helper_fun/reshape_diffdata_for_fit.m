%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Function to reshape diffusion data to a bval x n array for data fitting.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:
%   -   data
%   -   b-values
%
% Output:
%   -   diffdata
%       Reshaped diffusion data
%       The first dimension is the b-value and the second dimension 
%       contains the data. 
%   -   data_sz
%       The original array size (data_sz) is returned to allow reshaping 
%       of the data. 
%   -   Optional:
%       The b-value dimension can be returned (varargout{1}).
%       Permutation vector can be returned (varargout{2}).         
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [diffdata, data_sz, varargout] = reshape_diffdata_for_fit(data, bval)

%only for multi-voxel fit
if ~(size(data,2) == 1)
    if isempty(find(size(data, ndims(data))==length(bval), 1))
        error('Datasize and number of b-values are not matching!')
    end
    bdim = ndims(data);
    diffdata = permute(data, [bdim, 1:bdim-1 bdim+1:length(size(data))]);
    data_sz = size(diffdata);
    
    diffdata = reshape(diffdata, length(bval), []);
    diffdata = double(diffdata);
    
    if nargout > 2
        varargout{1} = bdim;
    end
    
    if nargout > 3
        %permutation vector
        varargout{2} = [bdim, 1:bdim-1 bdim+1:length(size(data))];
    end
else %single voxel case (e.g. simulations)
    diffdata = data;
    data_sz = size(data);
end