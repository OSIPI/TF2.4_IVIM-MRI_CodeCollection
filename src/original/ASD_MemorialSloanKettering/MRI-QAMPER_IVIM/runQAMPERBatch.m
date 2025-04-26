function runQAMPERBatch(batchStruct) %,csv_outdir)

batch_id = batchStruct.batch_id;
DWIstruct = batchStruct.DWIstruct;

% userCachePath = setUserCache;
% batchPath = [userCachePath filesep num2str(batch_id)];
batchPath = fullfile(batchStruct.currentNiiPath,num2str(batch_id));
csv_outdir = batchPath;

if ~exist(batchPath,'dir')
    mkdir(batchPath);
end

save(fullfile(batchPath,'batchStruct.mat'),'batchStruct');


roipath = [batchPath filesep 'roi'];
if ~exist(roipath)
    mkdir(roipath);
end
% disp(['Saving batch struct to ' batchStruct.currentNiiPath '...']);
% save(fullfile(batchStruct.currentNiiPath,['batchStruct_' num2str(batch_id) '.mat']),'batchStruct');


if DWIstruct.runDWI
    
    current_results = [];
    current_results.batch_id = batch_id;
    
    
    if ~isfield(DWIstruct,'Parallel')
        DWIstruct.Parallel = 1;
    end
    
    dwipath = [batchPath filesep 'dwi_ivim'];
    
    if ~exist(dwipath,'dir')
        mkdir(dwipath);
    end
    
    bval_arr = DWIstruct.bval_arr;
    bval_tf = DWIstruct.bval_tf;
    channels = DWIstruct.channels;
    N_avgs = DWIstruct.N_avgs;
    
    num_bvals = numel(find(bval_tf));
    
    bval_omit = find(~bval_tf);
        
    dwi_img = [];
    
    if ~iscell(DWIstruct.files{1})
        dwifile = deb(char(DWIstruct.files{1}(1,:)));
        copyfile(dwifile,dwipath);
        dwinii = nii_load(dwifile);
        dwi_img = dwinii.img;
    else
        for i = 1:numel(DWIstruct.files{1})
            dwifile = deb(DWIstruct.files{1}{i});
            copyfile(dwifile,dwipath);
            dwinii = nii_load(dwifile,1,DWIstruct.vols{i});
            dwi_img = cat(4,dwi_img,dwinii.img);
        end
    end
        
    current_results.dwinii = dwinii; current_results.pat_id = dwinii.fileprefix;
    current_results.dwinii.img = [];
    
    nanopt = DWIstruct.nanopt;
    smoothopt = DWIstruct.smoothopt;
    
    previewMode = DWIstruct.previewMode;
    previewAxes = DWIstruct.previewAxes;
    parallelFlag = DWIstruct.Parallel;
    
    if numel(bval_arr) ~= size(dwi_img,4)
        disp('Error, mismatched images and bvals');
    end
    
    %remove entries if unchecked
    if ~isempty(bval_omit)
        dwi_img(:,:,:,bval_omit) = [];
        bval_arr(bval_omit) = [];
    end
    
    if numel(bval_arr) ~= size(dwi_img,4)
        disp('Error, mismatched images and bvals');
    end
    
    dwi_series_num = '' ; %getSeriesNum(dwifile);
    
    roifile = DWIstruct.files{2};
    copyfile(roifile,dwipath);
    if ~isempty(roifile)
        roinii = nii_load(roifile);
        ROI = roinii.img;
    else
        ROI = ones(size(dwi_img(:,:,:,1)));
    end
    
    roi_series_num = '';
    
    roiType = DWIstruct.roiType; %getROIType(roifile);
    [~,fname,~] = fileparts(dwifile); 
    [~,fname,~] = fileparts(fname);
    save_prefix = [dwipath filesep fname];
    [bval_arr,dwi_img] = bvalOrderFix(bval_arr,dwi_img);
    
    if isfield(DWIstruct,'avgRepeatBvals')
        if DWIstruct.avgRepeatBvals
            [bval_arr,dwi_img] = avgRepeatBvalVols(bval_arr,dwi_img);
            num_bvals = numel(bval_arr);
        end
    end
%     
    % optional smoothing     
    if smoothopt
        dwi_smooth = zeros(size(dwi_img));
        smoothsize = 2*floor(DWIstruct.kernelsize/2)+1;
        for i = 1:num_bvals
            dwi_smooth(:,:,:,i) = smooth3(dwi_img(:,:,:,i),'gaussian',repmat(smoothsize,[1,3]));
        end
        dwi_img = dwi_smooth;
    end
    
    [dwi_arr, bval_arr, sigmadwi, SNR, noise] = dataPrepDWI(dwi_img,bval_arr,ROI,channels,N_avgs);
    
    testarr = dwi_arr - sigmadwi;
    numel(testarr < 0)
    
    poorSNRFlag = 0;
    
%     if SNR < 100
%         disp(['SNR is too low to continue (' num2str(SNR) ')']);
% %         H = warndlg('SNR in series is very low. Proceeding without proper noise handling');
%         sigmadwi = 0;
%         noise = 0;
%         poorSNRFlag = 1;
%     end
    
    %%%%% Matrix  methods for ADC and Kurtosis
    
    
    if DWIstruct.IVIM
        disp('Running IVIM');
        LB_IVIM = DWIstruct.LB_IVIM;
        UB_IVIM = DWIstruct.UB_IVIM;
        x0_IVIM = DWIstruct.x0_IVIM;
        
        [f_arr, D_arr, Dx_arr, s0_arr, fitted_dwi_arr, RSS, rms_val, chi, AIC, BIC, R_sq] = IVIM_standard_bcin(dwi_arr,bval_arr,sigmadwi,LB_IVIM,UB_IVIM,x0_IVIM,parallelFlag,previewMode,previewAxes);
        
        f_map = arr2map2(f_arr(:,1:end-1),ROI);
        nii = roinii;
        nii.img = f_map;
        f_filename = [save_prefix '_IVIM_f.nii.gz'];
        nii_save(nii,f_filename);
        
        D_map = arr2map2(D_arr(:,1:end-1),ROI);
        nii = roinii;
        nii.img = D_map;
        D_filename = [save_prefix '_IVIM_D.nii.gz'];
        nii_save(nii,D_filename);
        
        Dx_map = arr2map2(Dx_arr(:,1:end-1),ROI);
        nii = roinii;
        nii.img = Dx_map;
        Dx_filename = [save_prefix '_IVIM_Dx.nii.gz'];
        nii_save(nii,Dx_filename);
        
        s0_map = arr2map2(s0_arr(:,1:end-1),ROI);
        nii = roinii;
        nii.img = s0_map;
        s0_filename = [save_prefix '_IVIM_s0.nii.gz'];
        nii_save(nii,s0_filename);
        
        current_results = outputHNMPAResults(current_results,csv_outdir,'IVIM',roiType,{bval_arr,dwi_arr,[f_arr;D_arr;Dx_arr;s0_arr],fitted_dwi_arr,RSS,rms_val,chi,AIC,BIC,SNR,sigmadwi,LB_IVIM,UB_IVIM,R_sq,ROI});
%         updatePreviewAxes(previewAxes,bval_arr,dwi_arr(:,end),fitted_dwi_arr(:,end),{'b-value (s/mm^2)','signal (a.u.)','Mean fitted signal, IVIM'});
    end
end

