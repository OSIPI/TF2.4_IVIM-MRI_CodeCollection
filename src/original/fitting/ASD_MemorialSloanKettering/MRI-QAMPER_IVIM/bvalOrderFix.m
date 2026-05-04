function [bval_arr_new, dwi_img_new] = bvalOrderFix(bval_arr,dwi_img)

disp(['Reordering B0 to start of imaging sequence'])

if size(bval_arr,1) > size(bval_arr,2)
    bval_arr = bval_arr';
end
    
% bval_arr_new = [bval_arr(end) bval_arr(1:end-1)];
[bval_arr_new,idx] = sort(bval_arr);

% dwi_img_new = cat(4,dwi_img(:,:,:,end),dwi_img(:,:,:,1:end-1));
dwi_img_new = dwi_img(:,:,:,idx);

disp(bval_arr_new);