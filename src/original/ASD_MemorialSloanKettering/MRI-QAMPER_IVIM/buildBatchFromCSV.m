function batchStruct = buildBatchFromCSV(img_nii,roi_nii,qamper_mode,optionS)

batchStruct = [];
batchStruct.batch_id = num2str(now);

if strcmp(qamper_mode,'DWI')
    if isfield(optionS,'DWIstruct')
        DWIstruct = optionS.DWIstruct;
    end
    
    DWIstruct.runDWI = 1;
    [droot,fbase,ext] = fileparts(img_nii);
    batchStruct.currentNiiPath = fullfile(droot,'dwi_maps');

    
    if strcmp(ext,'.gz')
        [~,fbase,~] = fileparts(fbase);
    end
    
    bval_mat = fullfile(droot,[fbase '.mat']);
    
    T = load(bval_mat);
    
    DWIstruct.bval_arr  = T.bval_arr;
    
    img = nii_load(img_nii,1);
    vol_num = size(img.img,4);
    
    DWIstruct.files{1} = img_nii; DWIstruct.vols = vol_num;
    DWIstruct.files{2} = roi_nii;
    
    if ~isfield(DWIstruct,'ADC')
        DWIstruct.ADC = 0;
    end
    if ~isfield(DWIstruct,'IVIM')
        DWIstruct.IVIM = 0;
    end
    if ~isfield(DWIstruct,'NGIVIM')
        DWIstruct.NGIVIM = 0;
    end
    if ~isfield(DWIstruct,'Kurtosis')
        DWIstruct.Kurtosis = 0;
    end
    
    if ~isfield(DWIstruct,'roiType')
        DWIstruct.roiType = 'node';
    end
    
    if ~isfield(DWIstruct,'avgRepeatBvals')
        DWIstruct.avgRepeatBvals = 1;
    end
    
    if ~isfield(DWIstruct,'channels')
        DWIstruct.channels = 16; %
    end
    
    if ~isfield(DWIstruct, 'avgChannels')
        DWIstruct.avgChannels = 2; %
    end
    
    if DWIstruct.IVIM
        if ~isfield(DWIstruct,'LB_IVIM')
            IVIMBoundsTable = [{0} {0} {0} {0}; {0.5} {0.005} {0.5} {1.0}; {0.05} {0.0005} {0.01} {0.5}];
            
            DWIstruct.LB_IVIM = cell2mat(IVIMBoundsTable(1,:));
            DWIstruct.UB_IVIM = cell2mat(IVIMBoundsTable(2,:));
            DWIstruct.x0_IVIM = cell2mat(IVIMBoundsTable(3,:));
        end
    end
    if DWIstruct.NGIVIM
        if ~isfield(DWIstruct,'LB_NG')
            NGIVIMDefaultBoundsTable = [{0} {0} {0} {0} {0}; {0.5} {0.005} {0.1} {1.0} {2.0}; {0.05} {0.0005} {0.01} {1.0} {0.5}];
            
            DWIstruct.LB_NG = cell2mat(NGIVIMDefaultBoundsTable(1,:));
            DWIstruct.UB_NG = cell2mat(NGIVIMDefaultBoundsTable(2,:));
            DWIstruct.x0_NG = cell2mat(NGIVIMDefaultBoundsTable(3,:));
        end
    end    
    
    if ~isfield(DWIstruct,'Parallel')
        DWIstruct.Parallel = 1;
    end
    
    DWIstruct.bval_tf = ones(size(DWIstruct.bval_arr));
%     
%     DWIstruct.previewAxes = handles.fittingAxes;
%     
    DWIstruct.nanopt = 1;
    
    if ~isfield(DWIstruct,'kernelsize')
        DWIstruct.kernelsize = 1;
    end
    if ~isfield(DWIstruct,'smoothopt')
        DWIstruct.smoothopt = 1;
    end
    
    DWIstruct.previewMode = 0;
    DWIstruct.previewAxes = 0;

else
    DWIstruct.runDWI = 0;
end

batchStruct.DWIstruct = DWIstruct;
batchStruct.T1struct.runT1 = 0;
batchStruct.DCEstruct.runDCE = 0;
batchStruct.T2struct.runT2 = 0;