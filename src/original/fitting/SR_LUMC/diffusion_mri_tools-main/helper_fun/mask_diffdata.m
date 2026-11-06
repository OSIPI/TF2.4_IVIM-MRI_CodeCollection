%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Function to mask diffusion data. The masking is performed using a
% threshold on the mean (bval = min(bval)-data). The threshold can be set 
% manually using options.thr
%
% The masked data is reshaped to a 2-D array and has the size 
% (numel(bval), sels). 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [masked, sels] = mask_diffdata(data, bval, options)

arguments
    data
    bval
    options.thr {mustBeNumeric} = 0.01
end

%check data size
try
    bdim = find(size(data)==length(bval));
catch
    error('Datasize and number of b-values not matching!')
end

data = permute(data, [bdim, 1:bdim-1 bdim+1:length(size(data))]);

%print b-value used for masking
fprintf('Smallest b-value found and used for masking: b = %.2f \n', min(bval));

sels = mean(data(bval==min(bval),:),1) > options.thr*mean(data(bval==min(bval),:));
masked = data(:, sels);

