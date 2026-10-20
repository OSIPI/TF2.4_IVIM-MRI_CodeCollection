%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Function to perform a combined IVIM-DTI fit to data.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input values: 
%   - Diffusion data:   Array containing the data. Can be single voxel, a
%                       slice or volumentric data. The last dimension needs
%                       to match the number of b-values/diffusion direction 
%                       combinations. 
%
%   - b-values:         Vector containing the b-values
%
%   - diffusion 
%     directions:       nx3 vector containing the diffusion directions
%
% Options:
%   - initialguess:     Initial guess for D, f and Ds
%                       If not provided, default guess will be used [1, 
%                       0.2, 50] 
%                       For the diffusion tensor, the initial guess
%                       is set to D on the diagonal and 0 on the
%                       off-diagonal elements. 
%                       The initial guess for S0 is either 1 (if data is
%                       normalized) or the mean b-0 signal (if data is not
%                       normalized). 
%
%   - mask:             Default 1
%                       Background is masked by using a threshold cut-off
%                       value based on the minimum b-value data. 
%
%   - normalize:        Default 1
%                       Normalize data to the min (bval)-signal
%
%   - fit_method        Default 'free'
%                       'free', 'two_step', 'segmented', 'IVIM_correct'
%                       'free': The data is fitted to the full IVIM-DTI
%                       equation. 
%                       'two_step': First, the diffusion tensor and f are
%                       estimated using the high b-values only (b >= bcut).
%                       Then, Dtensor is kept constant and a bi-exponential
%                       fit is performed to estimate f and Ds. The
%                       previously calculated value for f is used as
%                       initial guess. 
%                       'segmented': A segmented IVIM-DTI fit is performed,
%                       using b = 0 and b >= bcut to estimate the diffusion
%                       tensor and f. D* ist not fitted with this method. 
%                       'IVIM_correct': First an IVIM fit is performed to
%                       obtain f and Ds. The data is then IVIM-corrected by
%                       substracting the IVIM component. A pure DTI fit is
%                       performed on the remaining data. 
%
%   - dti_constrained   Default 1
%                       Constrained fit for the diffusion tensor elements.
%                       The lower triangular matrix of a Cholesky
%                       decomposition of the diffusion tensor is fitted.
%                       After the fit, the diffusion tensor can be
%                       calculated. The fit boundaries differ from the
%                       unconstrained fit. 
%
%   - bcut:             Default 200
%                       Cut-off b-value for the two_step fit. The cut-off 
%                       b-value is included for analysis. 
%
% Output:
% Structure ivimdti_fit containing the following fields:
%   - S0 signal (fitted). In case of normalization S0 will be close to 1
%     for all voxels. 
%   - tensor: Diffusion tensor containing the 6 elements in the order
%             Dxx, Dyy, Dzz, Dxy, Dxz, Dyz
%   - perfusion fraction f
%   - pseudo-diffusion coefficient Ds
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function ivimdti_fit = fit_ivimdti(data, bval, diffdir, options)

arguments
    data 
    bval
    diffdir
    options.initialguess (3,1) = [1, 0.2, 50]
    options.mask {mustBeNumericOrLogical} = 1
    options.normalize {mustBeNumericOrLogical} = 1
    options.fit_method {mustBeMember(options.fit_method, {'free', 'two_step', 'segmented', 'IVIM_correct'})} = 'free'
    options.dti_constrained = 1;
    options.bcut {mustBeNumeric} = 200
end

%rearrange and reshape data to a bval x n array
[data, sz] =  reshape_diffdata_for_fit(data, bval);

%% optional: mask background voxels
%define datafit
if options.mask
    [datafit, sels] = mask_diffdata(data, bval);
else
    sels = true(1, size(data,2));
    datafit = data;
end

%% Set initial guess for fit, optional: normalize data
% x0 = [S0, Dxx, Dyy, Dzz, Dxy, Dxz, Dyz, f, D*]
if options.normalize
    datafit = norm_diffdata(datafit, bval);
    %set initial guess
    x0(1) = 1;
    x0(2:4) = options.initialguess(1);
    x0(5:7) = 0;
    x0(8:9) = options.initialguess(2:3);
else
    x0(1) = mean(datafit(bval==min(bval),:), 'all');
    x0(2:4) = options.initialguess(1);
    x0(5:7) = 0;
    x0(8:9) = options.initialguess(2:3);
end

%% scale b-value
if size(bval,1)==1
    bval = bval.';
end
bval = bval_scaling(bval);
options.bcut = bval_scaling(options.bcut);
%% set fit options
diffparams.bval = bval;
diffparams.diffdir = diffdir;

if options.dti_constrained
    %Fit Cholesky lower triangular matrix coefficients
    lb = [0   -9 -9 -9 -9 -9 -9 0   5+1e-5];
    ub = [inf  9  9  9  9  9  9 1 300];
else
    lb = [0   0 0 0 0 0 0 0   5];
    ub = [inf 3 3 3 3 3 3 1 300];
end
if options.normalize
    ub(1) = 10;
end

fitoptions = optimoptions('lsqcurvefit', 'Algorithm', 'levenberg-marquardt', ...
    'FunctionTolerance', 1e-10, 'maxIterations', 1000, 'OptimalityTolerance', 1e-10, ...
    'Display', 'off');

%initialize fit parameters
S0_fit = zeros(1, size(datafit, 2));
tensor_fit = zeros(6, size(datafit, 2));
f_fit = zeros(1, size(datafit, 2));
Ds_fit = zeros(1, size(datafit, 2));

%% perform fit
fprintf('Perform voxel-wise IVIM-DTI fit...\n');
tic;
fail = 0;
%Select fit method
switch options.fit_method
    
    %%%%%%%%%%%%%%%%%%%
    % free fit
    %%%%%%%%%%%%%%%%%%
    case 'free'
        %loop over voxels
        for v = 1:size(datafit,2)
            tmpdat = datafit(:,v);
            tmpdiff = diffparams;
            [tmpdat, tmpdiff] = remove_zeros(tmpdat, tmpdiff);
            [~,~,udiffdir] = unique(tmpdiff.diffdir(tmpdiff.bval>0,:), 'rows', 'stable');
            %only perform fit if at least 4 b-values and 6 unique diffusion
            %directions (without b0) are available
            if numel(unique(tmpdiff.bval)) > 3 && max(udiffdir) >= 6
                if options.dti_constrained
                    x = lsqcurvefit(@ivimdtifun_constr, x0, tmpdiff, tmpdat, lb, ub, fitoptions);
                    x(2:7) = lower_triangular2tensor(x(2:7));
                else
                    x = lsqcurvefit(@ivimdtifun, x0, tmpdiff, tmpdat, lb, ub, fitoptions);
                end
                S0_fit(:,v) = x(1);
                tensor_fit(:,v) = x(2:7);
                f_fit(:,v) = x(8);
                Ds_fit(:,v) = x(9);
            else
                fail = fail+1;
                x(1:9) = 0;
            end
        end
        
    %%%%%%%%%%%%%%%%%%%
    % two-step fit
    %%%%%%%%%%%%%%%%%%
    case 'two_step'
        %loop over voxels
        for v = 1:size(datafit,2)
            tmpdat = datafit(:,v);
            tmpdiff = diffparams;
            [tmpdat, tmpdiff] = remove_zeros(tmpdat, tmpdiff);
            [~,~,udiffdir] = unique(tmpdiff.diffdir(tmpdiff.bval>0,:), 'rows', 'stable');
            if v == 1
                lb_seg = lb([1 8 9]); ub_seg = ub([1 8 9]); x0_seg = x0([1 8 9]);
            end
            %only perform fit if at least 2 b-values and 6 unique diffusion
            %directions (without b0) are available
            if numel(unique(tmpdiff.bval)) > 1 && max(udiffdir) >= 6
                %First, fit diffusion tensor to b >= bcut
                if options.dti_constrained
                    [param(1:7)] = lsqcurvefit(@dtifun_constr, x0(1:7), calc_bmat(tmpdiff.bval(tmpdiff.bval>=options.bcut), tmpdiff.diffdir(tmpdiff.bval>=options.bcut,:)), ...
                        tmpdat(tmpdiff.bval>=options.bcut), lb(1:7), ub(1:7), fitoptions);
                    x(2:7) = lower_triangular2tensor(param(2:7));
                else
                    [param(1:7)] = lsqcurvefit(@dtifun, x0(1:7), calc_bmat(tmpdiff.bval(tmpdiff.bval>=options.bcut), tmpdiff.diffdir(tmpdiff.bval>=options.bcut,:)), ...
                        tmpdat(tmpdiff.bval>=options.bcut), lb(1:7), ub(1:7), fitoptions);
                    x(2:7) = param(2:7);
                end
                %Estimate f and use it as initial guess for IVIM fit
                S0_tmp = mean(tmpdat(tmpdiff.bval==min(tmpdiff.bval)));
                f_guess = 1 - (param(1) / S0_tmp);
                if f_guess < 0 || isnan(f_guess)
                    f_guess = 0;
                end
                x0_seg(2) = f_guess;
                
                %perform fit with fixed diffusion tensor
                tmpdiff.tensor = x(2:7);
                x([1,8:9]) = lsqcurvefit(@ivimdtifun, x0_seg, tmpdiff, tmpdat, lb_seg, ub_seg, fitoptions);
                
                
                S0_fit(:,v) = x(1);
                tensor_fit(:,v) = x(2:7);
                f_fit(:,v) = x(8);
                Ds_fit(:,v) = x(9);
            else
                fail = fail+1;
                x(1:9) = 0;
            end
        end
    
    %%%%%%%%%%%%%%%%%%%
    % segmented fit
    %%%%%%%%%%%%%%%%%%%  
    case 'segmented'
        %loop over voxels
        for v = 1:size(datafit,2)
            tmpdat = datafit(:,v);
            tmpdiff = diffparams;
            [tmpdat, tmpdiff] = remove_zeros(tmpdat, tmpdiff);
            [~,~,udiffdir] = unique(tmpdiff.diffdir(tmpdiff.bval>0,:),'rows', 'stable');
            %only perform fit if at least 2 b-values and 6 unique diffusion
            %directions (without b0) are available
            if numel(unique(tmpdiff.bval)) > 1 && max(udiffdir) >= 6
                if options.dti_constrained
                    x = lsqcurvefit(@ivimdtifun_seg_constr, x0(1:8), tmpdiff, tmpdat, lb(1:8), ub(1:8), fitoptions);
                    x(2:7) = lower_triangular2tensor(x(2:7));
                else
                    x = lsqcurvefit(@ivimdtifun_seg, x0(1:8), tmpdiff, tmpdat, lb(1:8), ub(1:8), fitoptions);
                end
                S0_fit(:,v) = x(1);
                tensor_fit(:,v) = x(2:7);
                f_fit(:,v) = x(8);
            else
                fail = fail+1;
                x(1:8) = 0;
            end
        end
        
    %%%%%%%%%%%%%%%%%%%
    % IVIM-correct DTI data
    %%%%%%%%%%%%%%%%%%%    
    % Based on QMRITools in Mathematica by Martijn Froeling
    case 'IVIM_correct'    
        %first, fit IVIM to mean signal to obtain D* to fix
        [mdat, ~] = mask_diffdata(data, bval);
        ivim_mean = fit_ivim(mean(mdat,2), bval, 'mask', 0, 'normalize', 1);
        fprintf('Fixed D* value obtained from mean fit: D* =  %d \n', ivim_mean.Ds);
        %then, fit IVIM (voxel-wise) to the signal with a fixed D* value (ivimfun)
        ivim_pars = fit_ivim(datafit.', bval, 'Ds_fix', ivim_mean.Ds, ...
            'mask', 0, 'normalize', 0);
        %IVIM-correct DTI signal by substracting IVIM component
        datafit_cor = datafit - ivim_pars.S0.*ivim_pars.f.*exp(-bval.*ivim_pars.Ds);
        %fit DTI to remaining DTI signal (dtifun)
        dti_pars = fit_dti(datafit_cor.', bval, diffdir, ...
            'constrained', options.dti_constrained, 'mask', 0, 'normalize', 0);      
        
        S0_fit = ivim_pars.S0;
        f_fit = ivim_pars.f;
        Ds_fit = ivim_pars.Ds;
        tensor_fit = dti_pars.D.';
end
fprintf('%d pixels were not fitted due to too much missing data. \n', fail);

fprintf('Fit performed in %.2f seconds. \n', toc);

%% rearrange output
sz_fit = sz(2:end);
if length(sz_fit)==1
    sz_fit = [1 sz_fit];
end

ivimdti_fit.S0       = zeros(sz_fit);
ivimdti_fit.S0(sels) = S0_fit;

ivimdti_fit.tensor           = squeeze(zeros([6, sz_fit]));
ivimdti_fit.tensor(:,sels)   = tensor_fit;
ivimdti_fit.tensor           = permute(ivimdti_fit.tensor, ...
    [2:length(size(ivimdti_fit.tensor)) 1]);

ivimdti_fit.f        = zeros(sz_fit);
ivimdti_fit.f(sels)  = f_fit;

if ~isequal(options.fit_method, 'segmented')
    ivimdti_fit.Ds       = zeros(sz_fit);
    ivimdti_fit.Ds(sels) = Ds_fit;
end

end
