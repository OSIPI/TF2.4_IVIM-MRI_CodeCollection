%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Function to calculate the IVIM signal given the IVIM parameters and
% b-values. 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input values:
%   - params:           vector containing the IVIM parameters in the order
%                       [S0, D, f, Ds]
%
%   - bval:             b-values
%                       In case of standard IVIM signal calculation, bval
%                       needs to be a vector containing the b-values. 
%                       In case of two-step IVIM fitting, bval is a
%                       structure with fields bval.D_fix (fixed value for
%                       the diffusion coefficient) and bval.bval (diffusion
%                       b-values). 
%
% The signal is calculated according to 
% S = S0*(f*exp(-b*Ds) + (1-f)*exp(-b*D))
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function signal = ivimfun(params, bval)

switch length(params)
    case 4
        %standard IVIM signal calculation
        S0 = params(1);
        D = params(2);
        f = params(3);
        Ds = params(4);
        
        if ~isnumeric(bval)
            error('No valid b-values provided')
        end
    case 3
        %two-step approach for IVIM fitting. D is fixed during the fit and is
        %provided as second argument in 'bval'.
        S0 = params(1);
        f = params(2);
        Ds = params(3);
        
        if isstruct(bval)
            D = bval.D_fix;            
            bval = bval.bval;
        else
            error('Incorrect input. No value for D is provided.')
        end
        
    case 2
        %two-step approach for IVIM fitting with fixed D*. 
        %D is also fixed during the fit (pre-calculated) and is provided as
        %second argument in 'bval'. 
        %D* is also stored in the structure 'bval'. 
        S0 = params(1); 
        f = params(2);
        if isstruct(bval)
            D = bval.D_fix;
            Ds = bval.Ds_fix;
            bval = bval.bval;
        else
            error('Incorrect input. No value for D or D* is provided.')
        end
    otherwise
        error('Wrong number of input parameters for IVIM equation')
end

%scale b-value
bval = bval_scaling(bval);

signal = S0*(f*exp(-bval*Ds) + (1-f)*exp(-bval*D));