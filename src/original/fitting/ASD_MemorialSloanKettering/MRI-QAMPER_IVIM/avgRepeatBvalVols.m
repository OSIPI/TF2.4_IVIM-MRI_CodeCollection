function [uniq_bval_arr, dwi_img_new] = avgRepeatBvalVols(bval_arr,dwi_img)

disp('Averaging multiple bval vols to increase signal')

[uniq_bval_arr,IA,~] = unique(bval_arr);

dwi_img_new = [];

for i = 1:numel(uniq_bval_arr)
    if i < numel(uniq_bval_arr)
        dwi_avg_vol = mean(dwi_img(:,:,:,IA(i):IA(i+1)-1),4);
    else
        dwi_avg_vol = mean(dwi_img(:,:,:,IA(i):end),4);
    end
    dwi_img_new = cat(4,dwi_img_new, dwi_avg_vol); 
end