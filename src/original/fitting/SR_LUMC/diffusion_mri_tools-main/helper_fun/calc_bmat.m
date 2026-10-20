%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Function to calculate the b-matrix from the b-values and the diffusion
% gradient directions. 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input: b-values (1 x n vector), diffusion directions (n x 3 vector)
% Output: 
%   bmatrix = bval.* [gx.^2, gy.^2, gz.^2, 2*gx*gy, 2*gx*gz, 2*gy*gz];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function bmat = calc_bmat(bval, diffdir)
    if size(bval,1)==1
        bval = bval.';
    end
    if size(diffdir,2) ~= 3
        diffdir = diffdir.';
    end
    %scale b-value
    bval = bval_scaling(bval);
    
    dir = [diffdir(:,1).^2, diffdir(:,2).^2, diffdir(:,3).^2, ...
        2*diffdir(:,1).*diffdir(:,2), ...
        2*diffdir(:,1).*diffdir(:,3), ...
        2*diffdir(:,2).*diffdir(:,3)];
    bmat = bval.*dir;
end