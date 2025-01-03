function [RSS,rms_val,chi,AIC,BIC,R_sq_mean] = qualityFit(Nb,Np,observed_arr,predicted_arr)

numVoxels = size(observed_arr,2);
RSS = zeros(numVoxels,1);
rms_val = zeros(numVoxels,1);
chi = zeros(numVoxels,1);
AIC = zeros(numVoxels,1);
BIC = zeros(numVoxels,1);

for i = 1:numVoxels
    observed_signal = observed_arr(:,i);
    predicted_signal = predicted_arr(:,i);
    
    signal_diff = observed_signal - predicted_signal;
    
    obs_mean = mean(observed_signal);
    SST(i) = sum((observed_signal - obs_mean).^2);
    
    RSS(i) = sum(signal_diff.^2);  %SSE
    
    R_sq(i) = 1 - (RSS(i)/SST(i));
    
    rms_val(i) = sqrt(RSS(i) / Nb);
    chi(i) = (sum(signal_diff.^2 ./ predicted_signal)) / Nb;
    AIC(i) = 2 * Np + Nb * log(RSS(i) / Nb) + (2 * Np * (Np + 1)) / (Nb - Np - 1); % AICc
    BIC(i) = Nb * log(RSS(i)/Nb) + Np * log(Nb);
end

disp(['mean Rsq ' num2str(mean(R_sq(find(R_sq > 0))))]);

R_sq_mean = mean(R_sq(find(R_sq > 0)));