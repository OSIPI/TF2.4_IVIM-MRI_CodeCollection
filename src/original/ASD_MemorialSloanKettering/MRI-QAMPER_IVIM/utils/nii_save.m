function nii_save(nii,niifile,vartype)
%author EML

if nargin > 2
    nii.hdr.dime.datatype = vartype;
    nii.hdr.dime.bitpix = vartype;
end

if isfield(nii, 'untouch')
    disp('Untouch option found');
    save_untouch_nii(nii,niifile);
else
    save_nii(nii,niifile);
end