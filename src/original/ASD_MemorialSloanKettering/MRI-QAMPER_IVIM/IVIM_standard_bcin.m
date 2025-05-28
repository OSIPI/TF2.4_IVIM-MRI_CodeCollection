function [f_arr, D_arr, Dx_arr, s0_arr, fitted_dwi_arr, RSS, rms_val, chi, AIC, BIC, R_sq] = IVIM_standard_bcin(dwi_arr,bval_arr,sigmadwi,LB0,UB0,x0in,parallelFlag,previewMode,previewAxes)



if size(bval_arr,2) < size(bval_arr,1)
    bval_arr = bval_arr';
end

numVoxels = size(dwi_arr,2);
print(numVoxels)
numBvals = length(bval_arr);

f_lb = 0;
% f_ub = 0.5;
f_ub = 1.0;
% f_ub = 0.75;
% D_lb = 0;
D_lb = 1e-6;
D_ub = 0.003;
% D_ub = 4e-3;
% Dx_lb = 0;
Dx_lb = 1e-6;
Dx_ub = 5e-2;
% Dx_ub = 0.5;
% Dx_ub = 300e-3;
s0_lb = 0;

% %Yousef bounds Jul 16 2018
d10 = 1e-3;
d20 = 1e-2;
f0 = 0.2;

% p0(i,:) = [    smax(i)       d10        d20    f0];
% lb(i,:) = [0.5*smax(i)  0.25*d10   0.25*d20   0.0];
% ub(i,:) = [2.0*smax(i)   4.0*d10    4.0*d20   1.0];

if nargin < 6
    x0in = [f0 d10 d20 1];  % entry for s (4th) is multiplicative factor, not absolute
    if nargin < 4
        %     LB = [f_lb D_lb Dx_lb s0_lb];
        %     UB = [f_ub D_ub Dx_ub 0];
        % entry for s (4th) is multiplicative factor, not absolute
        LB0 = [0 0.25*d10 0.25*d20 0.5];
        UB0 = [1 4*d10 4*d20 2];
    end
end

optimization_iterations = 4;

options = optimset('MaxFunEvals',400,'MaxIter',200, ...
    'TolFun',1e-6,'TolX',1e-6,'Display','off');

axisLabels = {'b-value (s/mm^2)','signal (a.u.)'};

tic
for i = 1:numVoxels
    if rem(i,1000) == 0
        toc
        disp(['Voxel ' num2str(i) ' of ' num2str(numVoxels)]);
        tic
    end
    s = dwi_arr(:,i);
    
    LB = LB0;
    UB = UB0;
    x0 = x0in;
    
    LB(end) = LB0(end)*max(s);
    UB(end) = UB0(end)*max(s);
    x0(end) = x0in(end)*max(s);
    %     disp(['Voxel ' num2str(i) ' of ' num2str(numVoxels)]);
    if ~parallelFlag
        for j = 1:optimization_iterations
            %         x0 = (LB + UB) / 2; %+ (UB - LB) * (rand - 0.5);
            [x_fit(j,:),resnorm(j),~,~,~] = lsqcurvefit(@(x,b)ivim_modeling_noise(x,b,sigmadwi), ...
                x0, bval_arr', s, LB, UB, options);
        end
    else
        parfor j = 1:optimization_iterations
            %         x0 = (LB + UB) / 2; %+ (UB - LB) * (rand - 0.5);
            [x_fit(j,:),resnorm(j),~,~,~] = lsqcurvefit(@(x,b)ivim_modeling_noise(x,b,sigmadwi), ...
                x0, bval_arr', s, LB, UB, options);
        end
    end
    [~,best_fit_idx] = min(resnorm);
    p = x_fit(best_fit_idx,:);
    
    fitted_dwi_arr(:,i) = ivim_modeling_noise(p,bval_arr',sigmadwi);
    f_arr(i) = p(1);
    D_arr(i) = p(2)*(1e3);
    if p(3) > 1e-5
        Dx_arr(i) = p(3)*(1e3);
    else
        Dx_arr(i) = 1e-5;
    end
    s0_arr(i) = p(4);
    
    if previewMode == 1 || previewMode == 100
        axisLabels{3} = ['IVIM Fit (Voxel # ' num2str(i) '/' num2str(numVoxels) ')'];
        if previewMode == 100 && i < 101
            updatePreviewAxes(previewAxes,bval_arr',s,fitted_dwi_arr(:,i),axisLabels);
        elseif previewMode == 1
            updatePreviewAxes(previewAxes,bval_arr',s,fitted_dwi_arr(:,i),axisLabels);
        end
     end
    
end

% Quantification of Fitting Quality
Nb = numel(bval_arr);
Np = numel(p);
observed_arr = dwi_arr;
predicted_arr = fitted_dwi_arr;
[RSS,rms_val,chi,AIC,BIC,R_sq] = qualityFit(Nb,Np,observed_arr,predicted_arr);
