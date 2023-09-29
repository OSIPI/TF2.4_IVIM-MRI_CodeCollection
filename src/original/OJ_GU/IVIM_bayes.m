function out = IVIM_bayes(Y,f,D,Dstar,S0,b,lim,n,rician,prior,burns,meanonly)
% out = IVIM_bayes(Y,f,D,Dstar,S0,b,N,lim,rician,prior,burns,meanonly)
%
% Input: 
%       Y:  a VxB matrix containing all image data where V is the number of 
%           voxels in the analysis ROI and B is the number of b-values
%       f:  a Vx1 vector containg the perfusion fraction parameters for all
%           voxels from a least squares fit
%       D:  a Vx1 vector containg the diffusion parameters for all voxels 
%           from a least squares fit
%       Dstar: a Vx1 vector containg the pseudo diffusion parameters for 
%           all voxels from a least squares fit
%       b:  a 1xB vector containing all b-values
%       lim:a 2x4 or 2x5 matrix with lower (1st row) and upper (2nd row) 
%           limits of all parameters in the order f,D,D*,S0,noisevar 
%       n:  number of iterations after "burn in" (default: 10000)
%       rician: scalar boolean. If true Rician noise distribution is used,
%           else Gaussian noise distribution is used (default: false)
%       prior: cell of size equal to number of estimated parameters
%           the cells should contain string equal to 'flat', 'reci' or
%           'lognorm' to define the prior disitribution for the
%           corresponding parameter (default: {'flat','flat','flat','flat','reci'}
%           'flat' = uniform, 'reci' = reciprocal, 'lognorm' = lognormal
%           lognormal is only available for D and D*
%       burns: number of burn-in steps (default: 1000)
%       meanonly: scalar boolean. if true, only the posterior mean and std
%           are estimated (substantially more memory efficient and
%           therefore faster) (default: false)
%
% Output:
%       out: a struct with the fields D, f, Dstar and S0(Vx1) containing the
%       voxelwise mean, median, mode and standard deviation of each parameter
%
% By Oscar Jalnefjord 2018-08-27
%
% If you use this function in research, please cite:
% Gustafsson et al. 2017 Impact of prior distributions and central tendency 
% measures on Bayesian intravoxel incoherent motion model fitting, MRM

%%%%%%%%%%%%%%%%%%
% Error handling %
%%%%%%%%%%%%%%%%%%
if ~isrow(b)
    b = b';
    if ~isrow(b)
        error('b must be a row vector');
    end
end
B = length(b);

if size(Y,2) ~= B
    Y = Y';
    if size(Y,2) ~= B
        error('Y must have same number of columns as b'); 
    end
end
V = size(Y,1);

if isvector(f) && length(f) > 1
    if ~isequal([V 1],size(f))
        f = f';
        if ~isequal([V 1],size(f))
            error('f must be a column vector with the same number of rows as Y or a scalar');
        end
    end
elseif isscalar(f)
    f = f*ones(V,1);
else
    error('f must be a column vector or a scalar');
end

if isvector(D) && length(D) > 1
    if ~isequal([V 1],size(D))
        D = D';
        if ~isequal([V 1],size(D))
            error('D must be a column vector with the same number of rows as Y or a scalar');
        end
    end
elseif isscalar(D)
    D = D*ones(V,1);
else
    error('D must be a column vector or a scalar');
end

if isvector(Dstar) && length(Dstar) > 1
    if ~isequal([V 1],size(Dstar))
        Dstar = Dstar';
        if ~isequal([V 1],size(Dstar))
            error('Dstar must be a column vector with the same number of rows as Y or a scalar');
        end
    end
elseif isscalar(Dstar)
    Dstar = Dstar*ones(V,1);
else
    error('Dstar must be a column vector or a scalar');
end

if isvector(S0) && length(S0) > 1
    if ~isequal([V 1],size(S0))
        S0 = S0';
        if ~isequal([V 1],size(S0))
            error('S0 must be a column vector with the same number of rows as Y or a scalar');
        end
    end
elseif isscalar(S0)
    S0 = S0*ones(V,1);
else
    error('S0 must be a column vector or a scalar');
end

% N = number of iterations
if nargin < 8
    n = 10000;
else
    if ~(isscalar(n) && n > 0 && (round(n) == n) && isreal(n))
        error('N must be a positive scalar integer');
    end
end
        
if nargin < 9
    rician = false; % use rician noise distribution
end

if nargin < 10
    prior = {'flat','flat','flat','flat','reci'}; % use flat prior distributions
end

% lim
if rician 
    limsz = [2,5];
else
    limsz = [2,4];
end
if ~isequal(size(lim),limsz)
    error('lim must be 2x%d',limsz(2));
end
if rician && ~isequal(size(lim),[2,5])
    error('noise variance must be estimated for Rician noise distribution');
end

% burn-in steps
if nargin < 11
    burns = 1000;
end

% mean only
if nargin < 12
    meanonly = false;
end

%%%%%%%%%%%%%%%%%%%%%%%%%
% Parameter preparation %
%%%%%%%%%%%%%%%%%%%%%%%%%
M = length(f);
nbs = length(b);

if rician
    % startvalue for variance
    is2 = nbs./sum((Y-repmat(S0,1,nbs).*((1-repmat(f,1,nbs)).*exp(-D*b) + ...
        repmat(f,1,nbs).*exp(-(D+Dstar)*b))).^2,2); %1/s2
end

if meanonly
    voxelsPerRun = V;
else
    voxelsPerRun = min(V,1500); % to avoid filling the memory for N > 40,000
end

if meanonly
    thetasum = zeros(M,4+rician);
    theta2sum = zeros(M,4+rician);
end

% burn-in parameters
burnUpdateInterval = 100;
burnUpdateFraction = 1/2;

%%%%%%%%%%%%%%%%%%%%%%%%
% Parameter estimation %
%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:ceil(M/voxelsPerRun)
    % partition voxels
    usedVoxels = (i-1)*voxelsPerRun + 1:min(i*voxelsPerRun,M);
    
    fpart = f(usedVoxels);
    Dpart = D(usedVoxels);
    Dstarpart = Dstar(usedVoxels);
    S0part = S0(usedVoxels);
    if rician
        is2part = is2(usedVoxels);
    end
    Ypart = Y(usedVoxels,:);
    Mpart = min(i*voxelsPerRun,M) - (i-1)*voxelsPerRun;
    
    % initialize parameter vector
    if meanonly
        theta = zeros(Mpart,4+rician,2);
    else
        theta = zeros(Mpart,4+rician,n+burns);
    end
    theta(:,1:4,1) = [fpart, Dpart, Dstarpart,S0part];
    if rician
        theta(:,5,1) = is2part;
    end
    

    % step length parameter
    w = zeros(Mpart,4+rician);
    w(:,1:4) = [fpart Dpart Dstarpart S0part]/10;
    if rician
        w(:,5) = 0.01*ones(Mpart,1);
    end

    N = zeros(Mpart,4+rician); % number of accepted samples
    
    % iterate for j = 2,3,...,n
    for j = 2:n + burns
        % initialize theta(j)
        if meanonly
            theta(:,:,2) = theta(:,:,1);
            thetanew = theta(:,:,2);
            thetaold = theta(:,:,1);
        else
            theta(:,:,j) = theta(:,:,j-1);
            thetanew = theta(:,:,j);
            thetaold = theta(:,:,j-1);
        end
        
        % sample each parameter
        for k = 1:4+rician
            % sample s and r and update
            s = thetaold(:,k) + randn(Mpart,1).*w(:,k);
            r = rand(Mpart,1); 

            thetas = thetanew;
            thetas(:,k) = s;
            alpha = acc_MH(thetas,thetanew,Ypart,b,lim,rician,prior{k});
            sample_ok = r < alpha;
            thetanew(sample_ok,k) = thetas(sample_ok,k);
            thetanew(~sample_ok,k) = thetaold(~sample_ok,k); % reject samples
            N(:,k) = N(:,k) + sample_ok;
        end
        
        % prepare for next iteration
        if meanonly
            theta(:,:,1) = thetanew;
        else
            theta(:,:,j) = thetanew;
        end

        % save parameter value after burn-in phase
        if meanonly && j > burns
            thetasum = thetasum + thetanew;
            theta2sum = theta2sum + thetanew.^2;
        end
        
        % adapt step length
        if j <= burns*burnUpdateFraction && mod(j,burnUpdateInterval) == 0
            w = w*(burnUpdateInterval+1)./(2*((burnUpdateInterval+1)-N));
            N = zeros(Mpart,4+rician);
        end


        % Display iteration every 500th iteration
        if ~mod(j,500) && j > burns
            disp(['Iterations: ' num2str(j-burns)]);
        elseif ~mod(j,100) && j < burns
            disp(['Burn in-steps: ' num2str(j)]);
        elseif j == burns
            disp(['Burn in complete: ' num2str(j)]);
        end
    end
    
    % Saves distribution measures
    if meanonly
        %mean
        out.f.mean(usedVoxels) = thetasum(:,1)/n;
        out.D.mean(usedVoxels) = thetasum(:,2)/n;
        out.Dstar.mean(usedVoxels) = thetasum(:,3)/n;
        out.S0.mean(usedVoxels) = thetasum(:,4)/n;
        
        % standard deviation
        out.f.std(usedVoxels) = sqrt(theta2sum(:,1)/n-(thetasum(:,1)/n).^2);
        out.D.std(usedVoxels) = sqrt(theta2sum(:,2)/n-(thetasum(:,2)/n).^2);
        out.Dstar.std(usedVoxels) = sqrt(theta2sum(:,3)/n-(thetasum(:,3)/n).^2);
        out.S0.std(usedVoxels) = sqrt(theta2sum(:,4)/n-(thetasum(:,4)/n).^2);
    else
        %mean
        out.f.mean(usedVoxels) = mean(squeeze(theta(:,1,burns + 1:n+burns)),2);
        out.D.mean(usedVoxels) = mean(squeeze(theta(:,2,burns + 1:n+burns)),2);
        out.Dstar.mean(usedVoxels) = mean(squeeze(theta(:,3,burns + 1:n+burns)),2);
        out.S0.mean(usedVoxels) = mean(squeeze(theta(:,4,burns + 1:n+burns)),2);

        %median
        out.f.median(usedVoxels) = median(squeeze(theta(:,1,burns + 1:n+burns)),2);
        out.D.median(usedVoxels) = median(squeeze(theta(:,2,burns + 1:n+burns)),2);
        out.Dstar.median(usedVoxels) = median(squeeze(theta(:,3,burns + 1:n+burns)),2);
        out.S0.median(usedVoxels) = median(squeeze(theta(:,4,burns + 1:n+burns)),2);

        % mode
        out.f.mode(usedVoxels) = halfSampleMode(squeeze(theta(:,1,burns + 1:n+burns)));
        out.D.mode(usedVoxels) = halfSampleMode(squeeze(theta(:,2,burns + 1:n+burns)));
        out.Dstar.mode(usedVoxels) = halfSampleMode(squeeze(theta(:,3,burns + 1:n+burns)));
        out.S0.mode(usedVoxels) = halfSampleMode(squeeze(theta(:,4,burns + 1:n+burns)));

        % standard deviation
        out.f.std(usedVoxels) = std(squeeze(theta(:,1,burns + 1:n+burns)),1,2);
        out.D.std(usedVoxels) = std(squeeze(theta(:,2,burns + 1:n+burns)),1,2);
        out.Dstar.std(usedVoxels) = std(squeeze(theta(:,3,burns + 1:n+burns)),1,2);
        out.S0.std(usedVoxels) = std(squeeze(theta(:,4,burns + 1:n+burns)),1,2);
    end
end


function alpha = acc_MH(thetas,thetaj,Y,b,lim,rician,prior)
% theta = [f, D, Dstar,S0,1/s2];
M = size(thetas,1);
N = length(b);

q = zeros(M,1);

% p(theta|lim)
pts = min((thetas >= repmat(lim(1,:),M,1)) & (thetas <= repmat(lim(2,:),M,1)),[],2);

% D < D*
pts = pts & (thetas(:,2) < thetas(:,3));

% signal model 
Ss = repmat(thetas(pts,4),1,N).*((1-repmat(thetas(pts,1),1,N)).*exp(-thetas(pts,2)*b) + repmat(thetas(pts,1),1,N).*exp(-thetas(pts,3)*b)); 
Sj = repmat(thetaj(pts,4),1,N).*((1-repmat(thetaj(pts,1),1,N)).*exp(-thetaj(pts,2)*b) + repmat(thetaj(pts,1),1,N).*exp(-thetaj(pts,3)*b)); 
ptsptj = double(pts);

if strcmp(prior,'reci')
    diffpar = find(thetas(1,:) ~= thetaj(1,:));
    ptsptj(pts) = thetaj(pts,diffpar)./thetas(pts,diffpar); % rejects samples outside the limits % ~pts already == 0

elseif strcmp(prior,'lognorm')
    diffpar = find(thetas(1,:) ~= thetaj(1,:));
    
%     ptsptj = pts;
    if diffpar == 2
        mu = -6;
        s = 1;
    elseif diffpar == 3
        mu = -3.5;
        s = 1;
    else
        error('lognorm prior not available'); % only for D and D*
    end
    ptsptj(pts) = lognormprior(thetas(pts,diffpar),mu,s)./lognormprior(thetaj(pts,diffpar),mu,s); % ~pts already == 0
elseif ~strcmp(prior,'flat')
    error('unknown prior');
end

if rician % rician noise distribution
    q(pts) = exp(-N*log(thetaj(pts,5))+sum(Y(pts,:).^2,2).*0.5.*thetaj(pts,5)+sum(Sj.^2,2).*0.5.*thetaj(pts,5)-sum(logIo(Y(pts,:).*Sj.*repmat(thetaj(pts,5),1,N)),2)...
        +N*log(thetas(pts,5))-sum(Y(pts,:).^2,2).*0.5.*thetas(pts,5)-sum(Ss.^2,2).*0.5.*thetas(pts,5)+sum(logIo(Y(pts,:).*Ss.*repmat(thetas(pts,5),1,N)),2)) .* ptsptj(pts);
else % gaussian noise distribution
    q(pts) = (sum((Y(pts,:)-Sj).^2,2)./sum((Y(pts,:)-Ss).^2,2)).^(N/2) .* ptsptj(pts);
end

alpha = min(1,q);


function p = lognormprior(x,mu,s)
p = 1./(s*sqrt(2*pi)*x).*exp(-(log(x)-mu).^2/(2*s^2));

function y = logIo(x)
% Returns the natural log of the 0th order modified Bessel function of first kind for an argument x
% Follows the exponential implementation of the Bessel function in Numerical Recipes, Ch. 6
%
% Translated to MATLAB from C++ from the FSL source code and vectorized for
% faster computations in MATLAB

b = abs(x);

y = zeros(size(x));

a1 = (x(b < 3.75)/3.75).^2;
a2 = 3.75./b(b >= 3.75);
y(b < 3.75) = log(1.0 + a1.*(3.5156229 + a1.*(3.0899424 + a1.*(1.2067492 + a1.*(0.2659732 + a1.*(0.0360768 + a1.*0.0045813))))));
y(b >= 3.75) = b(b >= 3.75) + log((0.39894228+a2.*(0.01328592+a2.*(0.00225319+a2.*(-0.00157565+a2.*(0.00916281+a2.*(-0.02057706+a2.*(0.02635537+a2.*(-0.01647633+a2.*0.00392377))))))))./sqrt(b(b>=3.75)));
















