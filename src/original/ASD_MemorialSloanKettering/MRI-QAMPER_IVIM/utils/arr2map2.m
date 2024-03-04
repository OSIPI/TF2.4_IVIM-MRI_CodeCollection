function map = arr2map2(arr,ROI)

dim = size(ROI);
idx = find(ROI);

%assume arr is a 3d map in 1xN form; if 4d timeseries with M timepoints,
%arr will be MXN form
%dim should have 3 dimensions

if size(arr,1) == 1 ||  size(arr,2) == 1
    if size(arr,1) < size(arr,2)
        arr = arr';
    end
end

% for some reason DWI/DCE had 1 more element in OPM so opted to remove last one
if size(arr, 1) > size(idx, 1)
    arr = arr(:,1:end-1);
end
if size(idx, 1) > size(arr, 1)
    idx = idx(1:end-1);
end

if numel(dim) < 3
    dim(3) = 1;
end

roisz = numel(find(ROI));

if size(arr,1) == roisz
    N = 2;
    tpts = size(arr,2);
else
    N = 1;
    tpts = size(arr,1);
end

% try
map = zeros(dim(1),dim(2),dim(3),size(arr,N));
% catch
%     arr = arr';
%     map = zeros(dim(1),dim(2),dim(3),size(arr,1));
% end

mapi = zeros(1,dim(1)*dim(2)*dim(3));

for i = 1:tpts
    if N == 1
        mapi(idx) = double(arr(i,:));
    elseif N == 2
        mapi(idx) = double(arr(:,i))';
    end
   map(:,:,:,i) = reshape(mapi,dim);

end