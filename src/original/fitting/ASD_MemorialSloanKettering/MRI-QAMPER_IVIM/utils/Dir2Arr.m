function [arr_names,sz]=Dir2Arr(srcpth,extns,omit,silentFlag) 
%Author: Eve LoCastro, Weill Cornell Medical College, Department of Radiology, IDEAL Imaging Lab, November 2011
%Dir2Arr sends dir search filenames to matlab array variable (like its pre- 
%decessor, DirToArr), but with extended functionality. srcpth, extns and  
%omit can be all be (cell) arrays of text. Returns text array with 
%spaced-padding at end 
 

if nargin < 4
    silentFlag = 0;
end
 
 
MSL=500; 
%  
if nargin > 2 && ~isempty(omit)
    omit_arr=Dir2Arr(srcpth,omit,'',silentFlag); 
    if isempty(omit_arr) 
        omit_list={}; 
    else 
        O=size(omit_arr,1); 
        for o=1:O 
            [~,f,e]=fileparts(omit_arr(o,:)); 
            omit_arr(o,:)=[f e blanks(MSL-length([f e]))]; 
        end 
        omit_list=cellstr(omit_arr); 
    end 
else 
    omit_list={}; 
end 

arr_names=[]; 
 
if ~iscellstr(srcpth), srcpth=cellstr(srcpth); end 
if ~iscellstr(extns), extns=cellstr(extns); end 
     
dupe_prevention={}; 
 
for p=1:length(srcpth) 
    read_dir=srcpth{p}; 
    for e=1:length(extns) 
        G=dir([read_dir filesep extns{e}]); 
        for g=1:length(G) 
            if ~ismember(G(g).name,omit_list) && ~strcmp(G(g).name(1),'.') && ~ismember(G(g).name,dupe_prevention) 
                fname=[read_dir filesep G(g).name]; 
                dupe_prevention{end+1}=G(g).name; 
                arr_names=[arr_names; fname blanks(MSL-length(fname))]; 
            end 
        end     
    end 
    %disp(p); 
end  
 
 
sz=size(arr_names,1); 

if ~silentFlag
    disp(['Dir2Arr: ' int2str(sz) ' entries.']);
end 
end 