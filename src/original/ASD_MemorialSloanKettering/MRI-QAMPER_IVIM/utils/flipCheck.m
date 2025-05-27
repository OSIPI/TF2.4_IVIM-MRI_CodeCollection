function LRflip = flipCheck(nii1,nii2)

% uses OtsuThreshold
% 
% nii1 = nii_load(img1,1);
% nii2 = nii_load(img2,1);

if nii1.hdr.dime.pixdim(1) == -nii2.hdr.dime.pixdim(1)
    LRflip = 1;
else
    LRflip = 0;
end

