function [pars,mask, gof] = IVIM_seg(Y,b,lim,blim,disp_prog)
    % function [pars,mask, gof] = IVIM_seg(Y,b,lim,blim,disp_prog)
    %
    % Function for monoexponential or stepwise biexponential fitting to
    % diffusion weighted MR data. The size of the input variable lim determines
    % the type of fit (monoexponential if size(lim) = [2 2], stepwise
    % biexponential if size(lim) = [2 4], estimation of D and f if 
    % size(lim) = [2 3])
    %
    % Input
    % - b is a column vector with B elements. It contains the b values
    % - Y is a BxV-matrix where V is the number of voxels. It contains the data
    % - lim is a 2x2-, 2x3 or 2x4-matrix where the first row gives the lower 
    %   limit and the second row the upper limits of the model parameters. The 
    %   order of the parameters is [D,S0], [D,S0,f] or [D,S0,f,D*] depending on
    %   the fit
    % - blim is a scalar that determines the b-values used in the first of the 
    %   fits (b == blim is included)
    % - disp_prog is a scalar boolean. If it is set to "true" the progress of
    %   the model fit is printed to the command window
    % 
    % Output
    % - pars is a struct with fields for all model parameters
    % - mask is a struct with fields D and Dstar specifying the voxels that 
    %   converged to a solution within the specified limits 
    % - gof is a struct with fields SSE (sum of squared errors) and R2
    %   (coefficient of determination)
    %
    % By Oscar Jalnefjord 2018-08-27
    % 
    % If you use this function in research, please cite:
    % Jalnefjord et al. 2018 Comparison of methods for estimation of the 
    % intravoxel incoherent motion (IVIM) diffusion coefficient (D) and 
    % perfusion fraction (f), MAGMA
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Error handling
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % b
    if ~iscolumn(b)
        if isrow(b)
            b = b';
        else
            error('b must be a vector');
        end
    end
    
    % Y
    if size(Y,1) ~= length(b)
        if size(Y,2) == length(b)
            Y = Y';
        else
            error('Dimensions of Y and b must agree');
        end
    end
    
    % lim
    if isequal(size(lim),[2,2]) % Limits [D,S0]
        step2 = false;  % fit monoexponential model
        estf = false;   
    elseif isequal(size(lim),[2,3]) % Limits [D,S0,f]
        step2 = false;
        estf = true;    % calculate f
        if ~any(b==0)
            error('b=0 is need to calculate f');
        end
    elseif isequal(size(lim),[2,4]) % Limits [D,S0,f,D*]
        step2 = true; % fit biexponential model
        estf = true;
    else
        error('lim must be 2x2, 2x3 or 2x4');
    end
    
    % blim
    if nargin < 4
        if step2 == false && estf == false
            blim = 0;
        else
            blim = 200;
        end
    end
    
    
    % Display progress
    if nargin < 5
        disp_prog = true;
    else
        if ~isscalar(disp_prog) || ~islogical(disp_prog)
            error('disp_prog must be a scalar boolean');
        end
    end
    
    if disp_prog
        fprintf('\nStarting optization process\n');
        fprintf('---------------------------\n');
    end
    
    
    % Common parameters
    optlim = 1e-6;
    n = size(Y,2);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    % Fitting D and A (S0)
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if disp_prog
        fprintf('Fitting D and A\n');
    end
    
    if estf
        % prepare variables for fitting at high b-values
        Yred = Y(b>=blim,:);
        bred = b(b>=blim);
        if length(bred) < 2
            error('blim is too large. At least two b-value must be >=blim')
        end
    else
        % monoexponential fit
        Yred = Y;
        bred = b;
    end
    
    % Mask to remove background 
    mask_back = sum(Y,1) > 0;
    if disp_prog
        fprintf('Discarding %d voxels as background\n',sum(~mask_back));
    end
    
    % Allocation
    D = zeros(1,n);
    maskD = mask_back;
    
    % Optimization
    [D(mask_back),maskD(mask_back)] = optimizeD(Yred(:,mask_back),bred,optlim,lim(:,1),disp_prog);
    
    % Calculates A based on D
    A = sum(Yred.*exp(-bred*D))./sum(exp(-2*bred*D));
    
    % Assures that A is within limits
    D(A < lim(1,2)) = lim(1,1);
    D(A > lim(2,2)) = lim(2,1);
    maskD(A < lim(1,2) | A > lim(2,2)) = false;
    A(A < lim(1,2)) = lim(1,2);
    A(A > lim(2,2)) = lim(2,2);
    
    % Calculate f
    if estf && ~step2
        if disp_prog
            fprintf('\nCalculating f\n');
        end
        f = 1-A./mean(Y(b==0,:),1);
        f(f < lim(1,3)) = lim(1,3);
        f(f > lim(2,3)) = lim(2,3);
        S0 = mean(Y(b==0,:),1);
    end
    
    
    if step2
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    % Fitting of D* and f (and S0)
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if disp_prog
            fprintf('\nFitting D* and f\n');
        end
    
        % calculation of f given D* and the previously estimated parameters (D and A)
        fcalc = @(Dstar,mask) sum(exp(-b*Dstar).*(Y(:,mask)-repmat(A(mask),length(b),1).*exp(-b*D(mask))))./sum(exp(-b*Dstar).*(Y(:,mask)-repmat(A(mask),length(b),1).*(exp(-b*D(mask))-exp(-b*Dstar))));
        
        % calculate possible range of f to remove voxels where f cannot
        % result in a fit within limits
        f1 = zeros(1,n);
        f1(mask_back) = fcalc(lim(1,4)*ones(1,sum(mask_back)),mask_back);
        f2 = zeros(1,n);
        f2(mask_back) = fcalc(lim(2,4)*ones(1,sum(mask_back)),mask_back);
    
        maskf = maskD & (((f1 > lim(1,3)) & (f1 < lim(2,3))) | ((f2 > lim(1,3)) & (f2 < lim(2,3))));
        if disp_prog
            fprintf('Discarding %d voxels due to f out of bounds\n',sum(maskD) - sum(maskf));
        end
    
        % Prepares output
        Dstar = zeros(1,n); 
        Dstar(f1 < lim(1,3)) = lim(1,4);
        Dstar(f2 > lim(2,3)) = lim(2,4);
        maskDstar = maskf;
    
        % Optimization
        if sum(maskf) > 0
            [Dstar(maskf),maskDstar(maskf)] = optimizeD(Y(:,maskf)-repmat(A(maskf),length(b),1).*exp(-b*D(maskf)),b,optlim,lim(:,4),disp_prog);
        end

        % Calculates f
        f = lim(1,1)*ones(1,n);
        f(f1 < lim(1,3)) = lim(1,3);
        f(f2 > lim(2,3)) = lim(2,3); 
        if sum(maskDstar) > 0
            f(maskDstar) = fcalc(Dstar(maskDstar),maskDstar);
        end
        
        % Checks for f out of bounds
        maskf = maskf & (f > lim(1,3)) & (f < lim(2,3));
        Dstar(f < lim(1,3)) = lim(1,4);
        Dstar(f > lim(2,3)) = lim(2,4);
        f(f < lim(1,3)) = lim(1,3);
        f(f > lim(2,3)) = lim(2,3);
        maskDstar = maskDstar & maskf;
    
        % Calculate S0
        S0 = Y(1,:);
        S0(f < .9) = A(f < .9)./(1 - f(f < .9)); % risk for Inf if f = 1 is used
    end
    
    if nargout == 3
        nbs = length(b);
        if step2
            gof.SSE = sum((Y - repmat(S0,nbs,1).*((1-repmat(f,nbs,1)).*exp(-b*D)+repmat(f,nbs,1).*exp(-b*(D+Dstar)))).^2,1);
        else
            gof.SSE = sum((Y - repmat(A,nbs,1).*exp(-b*D)).^2,1);
        end
        SStot = sum((Y-repmat(mean(Y,1),nbs,1)).^2,1);
        gof.R2 = 1-gof.SSE./SStot; % R2 can be negative
    end
    
    
    % Saves output variables
    pars.D = D;
    if step2
        pars.A = A;
    end
    mask.D = maskD;
    
    if estf
        pars.f = f;
        pars.S0 = S0;
        if step2
            pars.Dstar = Dstar;
            mask.Dstar = maskDstar;
        end
    else
        pars.S0 = A;
    end
    
    if disp_prog
        fprintf('\nDone!\n');
    end
    
    function [D,mask_background] = optimizeD(Y,b,optlim,Dlim,disp_prog)
    
    % Prepares variables
    n = size(Y,2);
    
    D = zeros(1,n);
    
    yb = Y .* repmat(b,1,n); % precalculated for speed
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Control if a minimum is within the interval
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % checks that all diff < 0 for Dlow
    Dlow = Dlim(1)*ones(1,n);
    difflow = Ddiff(Y,yb,b,Dlow,n);
    low_check = difflow < 0; % difflow must be < 0 if the optimum is within the interval
    
    % checks that all diff > 0 for Dhigh
    Dhigh = Dlim(2)*ones(1,n);
    diffhigh = Ddiff(Y,yb,b,Dhigh,n);
    high_check = diffhigh > 0; % diffhigh must be > 0 if the optimum if within the interval
    
    % sets parameter value with optimum out of bounds
    D(~low_check) = Dlim(1); % difflow > 0 means that the mimimum has been passed 
    D(~high_check) = Dlim(2); % diffhigh < 0 means that the minium is beyond the interval
    
    % Only the voxels with a possible minimum should be optimized
    mask = low_check & high_check;
    if disp_prog
        fprintf('Discarding %d voxels due to parameters out of bounds\n',sum(~mask));
    end
    
    % Saves a mask to know which voxels that has been optimized
    mask_background = mask;
    
    % Allocates all variables
    sz = size(Dlow);
    D_lin = zeros(sz);
    diff_lin = zeros(sz);
    D_mid = zeros(sz);
    diff_mid = zeros(sz);
    q1_lin = zeros(sz);
    q2_lin = zeros(sz);
    q1_mid = zeros(sz);
    q2_mid = zeros(sz);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Iterative method for finding the point where diff = 0 %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    k = 0;
    
    while any(mask) % Continues is there are voxels left to optimize
        % Assumes diff is linear within the search interval [Dlow Dhigh]
        D_lin(mask) = Dlow(mask) - difflow(mask) .* (Dhigh(mask) - Dlow(mask)) ./ (diffhigh(mask) - difflow(mask));
        % Calculates diff in the point of intersection given by the previous expression
        [diff_lin(mask),q1_lin(mask),q2_lin(mask)] = Ddiff(Y(:,mask),yb(:,mask),b,D_lin(mask),sum(mask));
        
        % As a protential speed up, the mean of Dlow and Dhigh is also calculated
        D_mid(mask) = (Dlow(mask) + Dhigh(mask))/2;
        [diff_mid(mask),q1_mid(mask),q2_mid(mask)] = Ddiff(Y(:,mask),yb(:,mask),b,D_mid(mask),sum(mask));
        
        % If diff < 0, then the point of intersection or mean is used as the
        % new Dlow. Only voxels with diff < 0 are updated at this step. Linear
        % interpolation or the mean is used depending of which method that
        % gives the smallest diff
        updatelow_lin = (diff_lin < 0) & ((diff_mid > 0) | ((D_lin > D_mid) & (diff_mid < 0)));
        updatelow_mid = (diff_mid < 0) & ((diff_lin > 0) | ((D_mid > D_lin) & (diff_lin < 0)));
        Dlow(updatelow_lin) = D_lin(updatelow_lin);
        Dlow(updatelow_mid) = D_mid(updatelow_mid);
        
        % If diff > 0, then the point of intersection or mean is used as the
        % new Dhigh. Only voxels with diff > 0 are updated at this step. 
        % Om diff �r > 0 ska sk�rningspunkten anv�ndas som ny Dhigh. Linear
        % interpolation or the mean is used depending of which method that
        % gives the smallest diff
        updatehigh_lin = (diff_lin > 0) & ((diff_mid < 0) | ((D_lin < D_mid) & (diff_mid > 0)));
        updatehigh_mid = (diff_mid > 0) & ((diff_lin < 0) | ((D_mid < D_lin) & (diff_lin > 0)));
        Dhigh(updatehigh_lin) = D_lin(updatehigh_lin);
        Dhigh(updatehigh_mid) = D_mid(updatehigh_mid);
        
        % Updates the mask to exclude voxels that fulfills the optimization
        % limit from the mask
        opt_lin = abs(1-q1_lin./q2_lin) < optlim;
        opt_mid = abs(1-q1_mid./q2_mid) < optlim;
        
        D(opt_lin) = D_lin(opt_lin);
        D(opt_mid) = D_mid(opt_mid); % Not optimal if both D_lin and D_mean fulfills the optimization limit, but has a small impact on the result as long as optlim is small
        
        % Updates the mask
        mask = mask & (~(opt_lin | opt_mid));
        
        % Calculates diff for the new bounds 
        if any(mask)
            difflow(mask) = Ddiff(Y(:,mask),yb(:,mask),b,Dlow(mask),sum(mask));
            diffhigh(mask) = Ddiff(Y(:,mask),yb(:,mask),b,Dhigh(mask),sum(mask));
        end
        
        k = k + 1;
        if disp_prog
            fprintf('Iteration %2d: %6d voxels left\n',k,sum(mask))
        end
    end
    
    function [diff,q1,q2] = Ddiff(Y,yb,b,D,n)
    q1 = sum(exp(-2*b*D),1) .* sum(yb.*exp(-b*D),1);
    q2 = sum(Y.*exp(-b*D),1).*sum(repmat(b,1,n).*exp(-2*b*D),1);
    diff = q1 - q2;
    