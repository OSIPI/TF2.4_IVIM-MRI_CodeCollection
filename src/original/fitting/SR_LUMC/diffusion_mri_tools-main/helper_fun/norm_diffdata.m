%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Function to normalize diffusion data. 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input: diffusion data, b-values
% Output: normalized diffusion data
% 
% The diffusion data is normalized to the mean b = 0 data. If no b = 0 data 
% is provided the data is normalized to the smallest b-value data. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function normalized = norm_diffdata(data, bval)

% check data
if ndims(data) > 2 %#ok<ISMAT>
    [data, sz, bdim_org, perm] = reshape_diffdata_for_fit(data, bval); 
end

% check if b0 data is provided
if min(bval) >= 1
    fprintf('No b = 0 data provided. Normalize to minimum b-value data: %.2f s/mm². \n',...
        min(bval))
    mS0 = mean(data((bval == min(bval)),:), 1);
else
    fprintf('Normalize data to b = %.2f data. \n', min(bval))
    mS0 = mean(data(bval<1, :), 1);
end

normalized = data ./ mS0;
normalized(isnan(normalized)) = 0;

if exist('sz', 'var')
    %reshape data to origianl size
    normalized = reshape(normalized, sz);
    bdim = find(size(normalized)==length(bval));
    if bdim ~= bdim_org
        %reverse permutation
        rp(perm) = 1:length(perm);
        normalized = permute(normalized, rp);
    end
end