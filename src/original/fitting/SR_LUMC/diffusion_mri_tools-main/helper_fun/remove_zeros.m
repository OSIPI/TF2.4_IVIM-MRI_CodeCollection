function [data, diffparams, varargout] = remove_zeros(data, diffparams, options)
arguments
    data
    diffparams
    options.bval = []
end

idx = data == 0;

if isstruct(diffparams)
    %IVIM-DTI
    diffparams.bval(idx) = [];
    diffparams.diffdir(idx,:) = [];
elseif isvector(diffparams)
    %IVIM, diffparams = bval
    diffparams(idx) = [];
else
    %DTI, diffparams = b-matrix
    diffparams(idx,:) = [];
end
data(idx) = [];

if ~isempty(options.bval)
    options.bval(idx) = [];
    varargout{1} = options.bval;
end

