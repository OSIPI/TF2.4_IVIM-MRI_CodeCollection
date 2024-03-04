
function batchStruct = buildBatchStruct(handles)

batchStruct.batch_id = handles.batch_id;


batchStruct.currentNiiPath = handles.outputDirectoryText.String;

qamperMode = handles.qamperMode;

if strcmp(qamperMode,'DWI')
    DWIstruct.runDWI = 1;
%     
%     DWIstruct.ADC = handles.ADCCheckBox.Value;
%     DWIstruct.IVIM = handles.IVIMCheckBox.Value;
%     DWIstruct.NGIVIM = handles.NGIVIMCheckBox.Value;
%     DWIstruct.Kurtosis = handles.KurtosisCheckBox.Value;
%     DWIstruct.Parallel = handles.ParallelButton_DWI.Value;

    fn = handles.fn;
% 
%     DWIstruct.files{1} = handles.inputDataText.String;
%     DWIstruct.files{2} = handles.inputROIText.String;
%         
    for i = 1:numel(fn)
        inputfile{i} = fn(i).filename; vol_num{i} = fn(i).filevolnum;
        DWIstruct.bval_arr(i) = fn(i).bval_arr;
    end
%     dwinputfilename = unique(inputfile); 
%     if numel(dwinputfilename) == 1
% %          = string(dwinputfilename);
%         dwinputfilename = char(dwinputfilename);
%     end
    DWIstruct.files{1} = inputfile; DWIstruct.vols = vol_num;
    DWIstruct.files{2} = handles.inputROIText.String;
    
%     [currentNiiPath,~,~] = fileparts(DWIstruct.files{1});
%     batchStruct.currentNiiPath = currentNiiPath;
    
    DWIstruct.ADC = handles.ADCCheckBox.Value;
    DWIstruct.IVIM = handles.IVIMCheckBox.Value;
    DWIstruct.NGIVIM = handles.NGIVIMCheckBox.Value;
    DWIstruct.Kurtosis = handles.DKICheckBox.Value;
    
    DWIstruct.roiType = handles.ROITypeDropDown.String{handles.ROITypeDropDown.Value};
    
    DWIstruct.avgRepeatBvals = handles.avgRepeatBvalsCheckBox.Value;
    
    % DWIstruct.channels = handles.channels;
    
    if isfield(handles, 'channels')
        DWIstruct.channels = handles.channels;
    else
        DWIstruct.channels = 24; %
    end
    
   % DWIstruct.avgChannels = handles.avgChannels;
   
   if isfield(handles, 'avgChannels')
       DWIstruct.avgChannels = handles.avgChannels;
   else
       DWIstruct.avgChannels = 24; %
   end
    
%     
%     presetValue = handles.presetPopupMenu.Value;
%     if presetValue == 1
%         presetValue = 2;
%     end
%
%     DWIstruct.preset = handles.presetPopupMenu.String{presetValue};

    if DWIstruct.IVIM
        DWIstruct.LB_IVIM = cell2mat(handles.IVIMUITable.Data(1,:));
        DWIstruct.UB_IVIM = cell2mat(handles.IVIMUITable.Data(2,:));
        DWIstruct.x0_IVIM = cell2mat(handles.IVIMUITable.Data(3,:));
    end
    if DWIstruct.NGIVIM
        DWIstruct.LB_NG = cell2mat(handles.NGIVIMUITable.Data(1,:));
        DWIstruct.UB_NG = cell2mat(handles.NGIVIMUITable.Data(2,:));
        DWIstruct.x0_NG = cell2mat(handles.NGIVIMUITable.Data(3,:));
    end
    
    if ~handles.parallelCheckBox.Value
        DWIstruct.Parallel = 0;

    else
        DWIstruct.Parallel = 1;
    end

    previewMode = handles.fitPreviewButtonGroup.SelectedObject.Tag;

    switch previewMode
        case 'previewNoneRadioButton'
            DWIstruct.previewMode = 0;
        case 'preview100RadioButton'
            DWIstruct.previewMode = 100;
        case 'previewFullRadioButton'
            DWIstruct.previewMode = 1;
    end
    
%     DWIstruct.bval_arr = handles.bval_arr;
    DWIstruct.bval_tf = ones(size(DWIstruct.bval_arr));
    
    DWIstruct.previewAxes = handles.fittingAxes;
    
    DWIstruct.nanopt = 1;
    
    if str2num(handles.smoothKernelEdit.String) > 0
        DWIstruct.smoothopt = 1;
        DWIstruct.kernelsize = str2num(handles.smoothKernelEdit.String);
    else
        DWIstruct.smoothopt = 0;
    end
    
%     if isempty(DWIstruct.bval_arr)
%         dwiniifile = DWIstruct.files{1};
%         [d,f,~] = fileparts(dwiniifile);
%         load([d filesep f(1:end-4) '.mat']);
%         DWIstruct.bval_arr = bval_arr;
%         if exist(bval_tf,'var')
%             DWIstruct.bval_tf = bval_tf;
%         else
%             DWIstruct.bval_tf = ones(size(bval_arr));
%         end
%     end
    
else
    DWIstruct.runDWI = 0;
end

if strcmp(qamperMode,'T1')
    T1struct.runT1 = 1;
    T1struct.AIFonly = 0; %handles.AIFcalcOnlyCheckBox.Value;
    
    fn = handles.fn;
    T1struct.FA = zeros(numel(fn),1);
    T1struct.TR = zeros(numel(fn),1);
    
    for i = 1:numel(fn)
        T1struct.files{i} = fn(i).filename;
        T1struct.FA(i) = fn(i).FA;
        T1struct.TR(i) = fn(i).TR;
    end
    
    T1struct.ROI = handles.inputROIText.String;
    
    if strcmp(handles.fitMethodButtonGroup.SelectedObject.String,'Non-Linear Fitting') && numel(fn) > 2
        T1struct.Nonlinear = 1;
        T1struct.LB = cell2mat(handles.T1NonLinearTable.Data(1,:));
        T1struct.UB = cell2mat(handles.T1NonLinearTable.Data(2,:));
    else
        T1struct.Nonlinear = 0;
    end
    
    switch handles.fitPreviewButtonGroup.SelectedObject.String
        case 'No Preview'
            T1struct.previewMode = 0;
        case 'First 200 Points Preview'
            T1struct.previewMode = 100;
        case 'Full Preview'
            T1struct.previewMode = 1;
    end                
    
    T1struct.Parallel = handles.parallelCheckBox.Value;
    T1struct.previewAxes = handles.fittingAxes;
    
else
    T1struct.runT1 = 0;
end

if strcmp(qamperMode,'T2')
    T2struct.runT2 = 1;
    
    fn = handles.fn;
    T2struct.TE = zeros(numel(fn),1);
    T2struct.TR = zeros(numel(fn),1);
    
    for i = 1:numel(fn)
        T2struct.files{i} = fn(i).filename;
        T2struct.TE(i) = fn(i).TE;
        T2struct.TR(i) = fn(i).TR;
    end
    
    T2struct.ROI = handles.inputROIText.String;
    
%     if strcmp(handles.fitMethodButtonGroup.SelectedObject.String,'Non-Linear Fitting') && numel(fn) > 2
%         T2struct.Nonlinear = 1;
%     else
%         T2struct.Nonlinear = 0;
%     end
    
    switch handles.fitPreviewButtonGroup.SelectedObject.String
        case 'No Preview'
            T2struct.previewMode = 0;
        case 'First 200 Points Preview'
            T2struct.previewMode = 100;
        case 'Full Preview'
            T2struct.previewMode = 1;
    end                
    
    T2struct.Parallel = handles.parallelCheckBox.Value;
    T2struct.previewAxes = handles.fittingAxes;
    
else
    T2struct.runT2 = 0;
end

if strcmp(qamperMode,'DCE')
    DCEstruct.runDCE = 1;
    
    fn = handles.fn;
    
    DCEstruct.AIF = handles.aifPopupMenu.String{handles.aifPopupMenu.Value};
    DCEstruct.Cpmat = handles.Cpmat;
    DCEstruct.Cp = handles.Cp;
    
    DCEstruct.SM2 = handles.SM2CheckBox.Value;
    DCEstruct.SM3 = handles.SM3CheckBox.Value;
    DCEstruct.SSM = handles.SSMCheckBox.Value;
    DCEstruct.Patlak = handles.PatlakCheckBox.Value;
    DCEstruct.CTUM = handles.CTUMCheckBox.Value;
    DCEstruct.CXM = handles.CXMCheckBox.Value;
    
    DCEstruct.roiType = handles.ROITypeDropDown.String{handles.ROITypeDropDown.Value};

%     dcemodelidx = handles.modelPopupMenu.Value;
%     dcemodel = handles.modelPopupMenu.String{dcemodelidx};
%     if dcemodelidx == 5 || dcemodelidx == 6
%         dcemodel = dcemodel(2:end);
%     end
    if DCEstruct.SM2 == 1
        eval('DCEstruct.SM2_LB = cell2mat(handles.SM2UITable.Data(1,:));');
        eval('DCEstruct.SM2_UB = cell2mat(handles.SM2UITable.Data(2,:));');
    end
    if DCEstruct.SM3 == 1
        eval('DCEstruct.SM3_LB = cell2mat(handles.SM3UITable.Data(1,:));');
        eval('DCEstruct.SM3_UB = cell2mat(handles.SM3UITable.Data(2,:));');
    end
    if DCEstruct.SSM == 1
        eval('DCEstruct.SSM_LB = cell2mat(handles.SSMUITable.Data(1,:));');
        eval('DCEstruct.SSM_UB = cell2mat(handles.SSMUITable.Data(2,:));');
    end
    if DCEstruct.CTUM == 1
        eval('DCEstruct.CTUM_LB = cell2mat(handles.CTUMUITable.Data(1,:));');
        eval('DCEstruct.CTUM_UB = cell2mat(handles.CTUMUITable.Data(2,:));');
    end
    if DCEstruct.CXM == 1
        eval('DCEstruct.CXM_LB = cell2mat(handles.CXMUITable.Data(1,:));');
        eval('DCEstruct.CXM_UB = cell2mat(handles.CXMUITable.Data(2,:));');
    end
%     if ~strcmp(dcemodel(1:3),'AIF') 
%         eval(['DCEstruct.' dcemodel ' = 1']);
%         if ~strcmp(dcemodel(1:3),'Pat')
%             eval(['DCEstruct.LB = cell2mat(handles. ' dcemodel 'UITable.Data(1,:));']);
%             eval(['DCEstruct.UB = cell2mat(handles. ' dcemodel 'UITable.Data(2,:));']);
%         end
%     end
    if handles.numItersPopupMenu.Value == 1
        DCEstruct.numIters = 4;
    else
        DCEstruct.numIters = str2num(handles.numItersPopupMenu.String{handles.numItersPopupMenu.Value});
    end
    
%     DCEstruct.T10 = handles.T10TextEdit.String;
    DCEstruct.T1 = '';
    DCEstruct.r1 = str2num(handles.RelaxivityTextEdit.String);
    if handles.constantT10CheckBox.Value
        DCEstruct.T1 = 'constant';
        DCEstruct.T10 = str2num(handles.T10TextEdit.String);
    else
        DCEstruct.T10file = handles.T10DataText.String;
        DCEstruct.m0file = handles.m0DataText.String;
    end
    
%     DCEstruct.Cpmat = handles.Cpmat;
    
%     if isempty(DCEstruct.acqDuration)
      DCEstruct.acqDuration = fn(end).t;
%     end
%     
%     if str2num(handles.smoothKernelEdit.String) > 0
%         DCEstruct.smoothopt = 1;
%         DCEstruct.kernelsize = str2num(handles.smoothKernelEdit.String);
%     else
        DCEstruct.smoothopt = 0;
%     end
    
    DCEstruct.Parallel = handles.parallelCheckBox.Value;
    
    %     DCEstruct.files{1} = handles.inputDataText.String;
    DCEstruct.t = zeros(1,numel(fn));
    DCEstruct.dcenii = fn(1).nii;
    for i = 1:numel(fn)
        inputfile{i} = fn(i).filename;
        DCEstruct.t(i) = fn(i).t;
    end
    dceinputfilename = unique(inputfile); 
    if numel(dceinputfilename) == 1
        dceinputfilename = char(dceinputfilename);
    end
    DCEstruct.files{1} = dceinputfilename;
    DCEstruct.files{2} = handles.inputROIText.String;
    
    DCEstruct.FA = fn(end).FA;
    DCEstruct.TR = fn(end).TR;
    
%     DCEstruct.t = handles.t;
    
    previewMode = handles.fitPreviewButtonGroup.SelectedObject.Tag;
    
    switch previewMode
        case 'previewNoneRadioButton'
            DCEstruct.previewMode = 0;
        case 'preview100RadioButton'
            DCEstruct.previewMode = 100;
        otherwise
            DCEstruct.previewMode = 1;
    end
    
    DCEstruct.previewAxes = handles.fittingAxes;
    DCEstruct.fn = fn;
else
    DCEstruct.runDCE = 0;
end

if strcmp(qamperMode,'T2')
    T2struct.runT2 = 1;
else
    T2struct.runT2 = 0;
end

batchStruct.DWIstruct = DWIstruct;
batchStruct.T1struct = T1struct;
batchStruct.DCEstruct = DCEstruct;
batchStruct.T2struct = T2struct;
