function arr = map2arr(map,roi)
%Author: Eve LoCastro, Weill Cornell Medical College, Department of Radiology, IDEAL Imaging Lab, November 2011
%assume map is a 3d map in LXMxNxT form if 4d timeseries with T timepoints,
%arr will be Tx(L*M*N) form

dim = size(map);
% if length(dim) < 4
%     idxlen = dim(1)*dim(2);
% else
% maplen = dim(1)*dim(2)*dim(3);
% end

if length(dim) < 4
    tpts = 1;
else
    tpts = dim(4);
end

% dotmap = map .* repmat(roi,1,1,1,tpts);

idx = find(roi);

arr = zeros(tpts,length(idx));

for i = 1:tpts
    tmpmap = map(:,:,:,i);
    arr(i,:) = tmpmap(idx)';
end