function nii = nii_load(niifile,untouchopt,img_idx)

%author EML

vartype = 16;

if nargin < 2 || isempty(untouchopt)
    untouchopt = 1;
end

if nargin < 3 || isempty(img_idx)
    img_idx = 0;
end

if ~untouchopt
    try
        if ~img_idx
            niiraw = load_nii(niifile);
        else
            niiraw = load_nii(niifile,img_idx);
        end
    catch
        disp(['Shear in image header detected, loading untouch nii ' niifile]);
        if ~img_idx
            niiraw = load_untouch_nii(niifile);
        else
            niiraw = load_untouch_nii(niifile,img_idx);
        end
    end
else
    if ~img_idx
        niiraw = load_untouch_nii(niifile);
        idxstr = '';
    else
        niiraw = load_untouch_nii(niifile,img_idx);
        idxstr = [' vol ' num2str(img_idx)];
    end
    disp(['option selected loading untouched nii ' niifile idxstr]);
end

nii = niiraw;
nii.hdr.dime.datatype = vartype;
nii.hdr.dime.bitpix = vartype;

nii.img = double(niiraw.img);