function [dwi_arr, bval_arr, sigmadwi, SNR, noise] = dataPrepDWI(dwi_img,bval_arr,ROI,channels,N_avgs)

bgMask = zeros(size(dwi_img,1),size(dwi_img,2));
bgMask(10:30,10:30) = 1;
% bgidx = sub2ind(size(bgMask),10:30,10:30);
bgidx = find(bgMask);

maskidx = find(ROI);

dwi_arr = map2arr(dwi_img,ROI);
dwi_arr(:,end+1) = mean(dwi_arr,2);

noise_square = bgMask.*dwi_img(:,:,1,1);
noise_arr = noise_square(bgidx);noise_arr;

%n = 8; % need to update with new protocol
%navg = 4;
n = channels;
navg = N_avgs;

% if data were converted to nifti before conv_panel started to extract
% channels and N_avgs, then those numbers default to this
if isempty(n)
    n = 16;
    navg = 2;
end

sigmadwi = sqrt(navg*norm(noise_arr(:))^2/length(noise_arr(:)))/sqrt(2*n);
noise = mean(noise_arr);
SNR = mean(dwi_arr(1,:))/sigmadwi;