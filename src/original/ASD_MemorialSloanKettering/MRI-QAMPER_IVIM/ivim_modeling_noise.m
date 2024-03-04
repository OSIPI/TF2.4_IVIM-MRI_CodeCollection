function s = ivim_modeling_noise(x,b,sigma)
%IVIM analysis
%s = s0*((1-f)exp(-b.D) + f.exp(- b(D + Dstar)))
%x = [f,D,Dstar];


f = x(1);
D = x(2);
Dstar = x(3);
s0 = x(4);

s = s0*((1-f)*exp(-b*D) + f*exp(-b*Dstar));
% n = 8;
% navg = 4;
% s = sqrt(s.^2 + 2*n*sigma.^2/navg);

s = sqrt(s.^2 + sigma.^2);
