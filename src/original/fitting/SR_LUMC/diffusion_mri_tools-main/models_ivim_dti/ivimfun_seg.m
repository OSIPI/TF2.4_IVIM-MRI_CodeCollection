%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Function to calculate the segmented IVIM signal given the IVIM parameters 
% and b-values using the segmented IVIM equation. 
%
% The pseudo-diffusion coefficient D* is not calculated with this method. 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input values:
%   - params:           vector containing the segmented IVIM parameters in 
%                       the order [S0, D, f]
%
%   - bval:             b-values
%                       bval is a vector containing the b-values. 
%
% The signal is calculated according to 
% S = S0*(f*delta_fun(bval) + (1-f)*exp(-b*D))
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function signal = ivimfun_seg(params, bval)

if length(params) < 3
    error('Wrong number of input parameters for segmented IVIM equation')
end
if length(params) > 3
    warning('Too many input parameters provided, additional parameters are ignored')
end

S0 = params(1);
D = params(2);
f = params(3);

%scale b-value
bval = bval_scaling(bval);

signal = S0*(f*delta_fun(bval) + (1-f)*exp(-bval*D));