function [data, diffparams, varargout] = remove_zeros(data, diffparams, options)
arguments
    data
    diffparams
    options.bval = []
end

if isstruct(diffparams)
    %IVIM-DTI
    diffparams.bval(data==0) = [];
    diffparams.diffdir(data==0,:) = [];
elseif isvector(diffparams)
    %IVIM, diffparams = bval
    diffparams(data==0) = [];
else
    %DTI, diffparams = b-matrix
    diffparams(data==0,:) = [];
end
data(data==0) = [];

if ~isempty(options.bval)
    options.bval(data==0) = [];
    varargout{1} = options.bval;
end

