%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Function to perform an DTI fit to data. 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input values: 
%   - diffusion data:   Array containing the data. Can be single voxel, a
%                       slice or volumentric data. The last dimension needs
%                       to match the number of b-value-diffusion direction 
%                       combinations. 
%
%   - b-values:         Vector containing the b-values 
%
%   - diffusion 
%     directions:       nx3 vector containing the n diffusion directions. 
%
% Options: 
%   - initialguess:     Initial guess for D
%                       If not provided, default guess will be used (1). 
%                       For the diffusion tensor, the initial guess
%                       is set to D on the diagonal and 0 on the
%                       off-diagonal elements. 
%                       The initial guess for S0 is either 1 (if data is
%                       normalized) or the mean b-0 signal (if data is not
%                       normalized).
%
%   - mask:             Default 1
%                       Background is masked using a threshold cut-off 
%                       value 
%
%   - constrained:      Default 1
%                       Constrained: fit the lower triangular matrix of a
%                       Cholesky decomposition. The fit boundaries differ
%                       from the unconstrained fit. After the tensor fit,
%                       the diffusion tensor needs to be calculated from the
%                       lower triangular matrix. 
%                       Unconstrained: fit the 6 diffusion tensor elements
%                       directly. 
%
%   - normalized:       Default 1
%                       Normalize data to min(bval)-signal
%
% Output:
% Structure dti_fit containing the following fields:
%   - S0 signal (fitted). In case of normalization S0 will be close to 1
%     for all voxels. 
%   - diffusion tensor D
%     vector containing the 6 elements of the diffusion tensor. 
%     Order of the tensor elements is:
%     Dxx, Dyy, Dzz, Dxy, Dxz, Dyz. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function dti_fit = fit_dti(data, bval, diffdir, options)

arguments
    data
    bval
    diffdir
    options.initialguess (1,1) = 1
    options.mask {mustBeNumericOrLogical} = 1
    options.constrained {mustBeNumericOrLogical} = 1
    options.normalize {mustBeNumericOrLogical} = 1
end

%rearrange and reshape data to a bval x n array
[data, sz] = reshape_diffdata_for_fit(data, bval);

if size(bval,1)==1
    bval = bval.';
end
%% optional: mask background voxels
%define datafit
if options.mask
    [datafit, sels] = mask_diffdata(data, bval);
else
    sels = true(1,size(data,2));
    datafit = data;
end

%% Set initial guess for fit, optional: normalize data
if options.normalize
    datafit = norm_diffdata(datafit, bval);
    %set initial guess
    x0(1) = 1;
    x0(2:4) = options.initialguess;
    x0(5:7) = 0;
else
    x0 = mean(datafit(bval==min(bval),:), 'all');
    x0(2:4) = options.initialguess;
    x0(5:7) = 0;
end

%% calculate b-matrix
bval = bval_scaling(bval);
bmat = calc_bmat(bval, diffdir);

%% set fit options
if options.constrained
    %Fit Cholesky lower triangular matrix of diffusion tensor
    lb = [0   -9 -9 -9 -9 -9 -9];
    ub = [inf  9  9  9  9  9  9];
else
    lb = [0    0  0  0  0  0  0];
    ub = [inf  3  3  3  3  3  3];
end

if options.normalize
    ub(1) = 10;
end
fitoptions = optimoptions('lsqcurvefit', 'Algorithm', 'levenberg-marquardt', ...
    'FunctionTolerance', 1e-10, 'MaxIterations', 1000, 'OptimalityTolerance', 1e-10, 'StepTolerance', 1e-10, ...
    'Display', 'off');

%initialize diffusion tensor and S0
tensor = zeros(6, size(datafit,2));
S0_fit = zeros(1, size(datafit,2));

%% perform fit
fprintf('Perform voxel-wise DTI fit...\n');
tic; 
fail = 0;
for v = 1:size(datafit,2)
    tmpdat = datafit(:,v);
    tmpbmat = bmat;
    [tmpdat, tmpbmat] = remove_zeros(tmpdat, tmpbmat);
    [~,~,udiffdir] = unique(tmpbmat, 'rows', 'stable');
    %only perform fit if at least 7 unique diffusion directions 
    % (scans) are used
    if max(udiffdir) > 6
        if options.constrained
            t = lsqcurvefit(@dtifun_constr, x0, tmpbmat, tmpdat, lb, ub, fitoptions);
            % Calculate tensor elements from Cholesky lower triangular matrix
            t(2:7) = lower_triangular2tensor(t(2:7));
        else
            t = lsqcurvefit(@dtifun, x0, tmpbmat, tmpdat, lb, ub, fitoptions);
        end
        
        tensor(:,v) = t(2:7);
        S0_fit(v) = t(1);
    else
        fail = fail + 1;
        t(1:7) = 0;
    end
end
fprintf('%d pixels were not fitted due to too much missing data. \n', fail);

fprintf('Fit performed in %.2f seconds. \n', toc);

%% rearrange output
sz_fit = sz(2:end);
if length(sz_fit)==1
    sz_fit = [1 sz_fit];
end

dti_fit.S0 = zeros(sz_fit);
dti_fit.S0(sels) = S0_fit;

dti_fit.D = squeeze(zeros([6, sz_fit]));
dti_fit.D(:,sels) = tensor;
dti_fit.D = permute(dti_fit.D, [2:length(size(dti_fit.D)) 1]);
end
