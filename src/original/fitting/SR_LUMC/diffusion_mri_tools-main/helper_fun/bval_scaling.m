%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Function to scale the b-value to the unit 10^-3 s/mm² to match with the
% IVIM/DTI parameter units. 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input: b-value (array/list)
% Output: scaled b-value
%
% If the b-value is scaled correctly already, the original b-value is
% returned.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function bval_scaled = bval_scaling(bval)

if max(bval) > 100
    bval_scaled = bval./1000;
else
    bval_scaled = bval;
end