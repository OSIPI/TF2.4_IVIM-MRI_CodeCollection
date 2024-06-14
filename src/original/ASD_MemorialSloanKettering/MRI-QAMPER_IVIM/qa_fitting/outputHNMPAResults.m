function current_results_out = outputHNMPAResults(current_results,csv_outdir,modality,roiType,resultsCell)


current_results_out = current_results;

struct_string = [modality '_' roiType];

%         current_results = outputHNMPAResults(current_results,'IVIM',roiType,{bval_arr,dwi_arr,[f_arr;D_arr;Dx_arr;s0_arr],fitted_dwi_arr,RSS,rms_val,chi,AIC,BIC});
%                 current_results = outputHNMPAResults(current_results,'IVIM',roiType,{bval_arr,dwi_arr,[f_arr;D_arr;Dx_arr;s0_arr],fitted_dwi_arr,RSS,rms_val,chi,AIC,BIC,SNR,sigma,LB_IVIM,UB_IVIM,ROI});
eval(['current_results_out.' struct_string '_b_value_fit = resultsCell{1};']); bval_arr = resultsCell{1};
eval(['current_results_out.' struct_string '_dwi_data = resultsCell{2};']);

params_out = resultsCell{3};

eval(['current_results_out.' struct_string '_params_f = params_out(1,:);']); f = params_out(1,:);
eval(['current_results_out.' struct_string '_features_f = returnFeatures(params_out(1,:));']);

eval(['current_results_out.' struct_string '_params_D = params_out(2,:);']); D = params_out(2,:);
eval(['current_results_out.' struct_string '_features_D = returnFeatures(params_out(2,:));']);

eval(['current_results_out.' struct_string '_params_Dx = params_out(3,:);']); Dx = params_out(3,:);
eval(['current_results_out.' struct_string '_features_Dx = returnFeatures(params_out(3,:));']);

eval(['current_results_out.' struct_string '_params_s0 = params_out(4,:);']);
eval(['current_results_out.' struct_string '_features_s0 = returnFeatures(params_out(4,:));']);

eval(['current_results_out.' struct_string '_fitted_dwi_data1 = resultsCell{4};']);
%Fit quality
eval(['current_results_out.' struct_string '_quality_RSS = resultsCell{5};']); RSS = resultsCell{5};
eval(['current_results_out.' struct_string '_quality_rms_val = resultsCell{6};']);
eval(['current_results_out.' struct_string '_quality_chi = resultsCell{7};']);
eval(['current_results_out.' struct_string '_quality_AIC = resultsCell{8};']);
eval(['current_results_out.' struct_string '_quality_BIC = resultsCell{9};']);
eval(['current_results_out.' struct_string '_quality_SNR = resultsCell{10};']); SNR = resultsCell{10};
eval(['current_results_out.' struct_string '_quality_sigma = resultsCell{11};']); sigmadwi = resultsCell{11};
eval(['current_results_out.' struct_string '_LB_IVIM = resultsCell{12};']); LB_IVIM = resultsCell{12};
eval(['current_results_out.' struct_string '_UB_IVIM = resultsCell{13};']); UB_IVIM = resultsCell{13};
eval(['current_results_out.' struct_string '_quality_Rsq = resultsCell{end-1};']); R_sq = resultsCell{end-1};
eval(['current_results_out.' struct_string '_ROI_map = resultsCell{end};']); ROI = resultsCell{end};

outstringheader='pat_ID,analysis_batch,tissue,D_Mean,D_Median,D_StDev,D_Skewness,D_Avg_Signal,D*_Mean,D*_Median,D*_StDev,D*_Skewness,D*_Avg_Signal,f_Mean,f_Median,f_StDev,f_Skewness,f_Avg_Signal,volume,voxels,RSS,b-values,SNR,sigma,R_sq,D_Kurtosis,D*_Kurtosis,f_Kurtosis';

outcsv = [csv_outdir filesep 'dwi_IVIM.csv']; voxdim = current_results.dwinii.hdr.dime.pixdim(2) * current_results.dwinii.hdr.dime.pixdim(3) * current_results.dwinii.hdr.dime.pixdim(4);
voxnum = numel(find(ROI));
outstring = [current_results.pat_id ',' num2str(current_results.batch_id) ',' roiType ',' num2str(mean(D(1:end-1))) ',' num2str(median(D(1:end-1))) ',' num2str(std(D(1:end-1))) ',' num2str(skewness(D(1:end-1))) ',' num2str(D(end)) ','  ...
    num2str(mean(Dx(1:end-1))) ',' num2str(median(Dx(1:end-1))) ',' num2str(std(Dx(1:end-1))) ',' num2str(skewness(Dx(1:end-1))) ',' num2str(Dx(end)) ','  ...
    num2str(mean(f(1:end-1))) ',' num2str(median(f(1:end-1))) ',' num2str(std(f(1:end-1))) ',' num2str(skewness(f(1:end-1))) ',' num2str(f(end)) ','  ...
    num2str(voxdim * voxnum) ',' num2str(voxnum) ',' num2str(mean(RSS)) ',' num2str(numel(bval_arr)) ',' num2str(SNR) ',' num2str(sigmadwi) ',' num2str(R_sq) ',' num2str(kurtosis(D(1:end-1))) ',' num2str(kurtosis(Dx(1:end-1))) ',' num2str(kurtosis(f(1:end-1)))];
if ~exist(outcsv,'file')
    fid = fopen(outcsv,'w');
    fprintf(fid,'%s\n',outstringheader);
else
    fid = fopen(outcsv,'a');
end
fprintf(fid,'%s\n',outstring);
fclose(fid);
eval(['current_results_out.' struct_string '_CSV = outstring ;']);
