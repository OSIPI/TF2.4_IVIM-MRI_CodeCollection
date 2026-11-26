%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This function calculates the (diffusion) tensor from the given Cholesky
% lower triangular matrix (obtained by Cholesky decomposition). 
% Input is a n x 6 vector containing the matrix elements. 
%
% Output are the tensor elements in the order Dxx, Dyy, Dzz, Dxy, Dxz, Dyz
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function cd = lower_triangular2tensor(r)
    % Cholesky decomposition
    % cd = (r1   0   0  )   (r1  r4  r5 )
    %      (r4   r2  0  ) * (0   r2  r6 )
    %      (r5   r6  r3 )   (0   0   r3 )
    %
    %
    %   =  ( r1^2    r1*r4       r1*r5           )
    %      (         r4^2+r2^2   r4*r5 + r2*r6   )
    %      (                     r5^2+r6^2+r3^2  )

%check dimension
if ndims(r) ~= 2 %#ok<ISMAT>
    error('wrong matrix size')
end
if isempty(find(size(r)==6, 1))
    error('no valid tensor provided')
end
    
if min(size(r))==1
    %single voxel
    cd = [r(1).^2, r(4).^2+r(2).^2, r(5).^2+r(6).^2+r(3).^2, ...
        r(1).*r(4), r(1).*r(5), r(4).*r(5)+r(2).*r(6)];
else
    %multiple voxels
    if size(r,1) ~= 6
        r = r.';
    end
    cd = [r(1,:).^2; r(4,:).^2+r(2,:).^2; r(5,:).^2+r(6,:).^2+r(3,:).^2; ...
        r(1,:).*r(4,:); r(1,:).*r(5,:); r(4,:).*r(5,:)+r(2,:).*r(6,:)];
end