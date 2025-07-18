function write_param_txt(param_map,ROI,txtfilename,transformscale)
%
% author EML
% write parameter values in txt file for use in COMSOL
if nargin < 4 || isempty(transformscale)
    transformscale = 1;
end

idx = find(ROI);

if size(param_map,3) > 1
    [Vx,Vy,Vz] = ind2sub(size(param_map),idx);
%     formatspec = '%5.1f \t %5.1f \t %5.1f \t %8.8f \n';
    formatspec = '%5.1f \t %5.1f \t %5.1f \t %e \n';
    interparr = [transformscale*Vx transformscale*Vy transformscale*Vz param_map(idx)]';
else
    [Vx,Vy] = ind2sub(size(param_map),idx);
    formatspec = '%5.1f \t %5.1f \t %8.8f \n';
    interparr = [transformscale*Vx transformscale*Vy param_map(idx)]';
end

fid = fopen(txtfilename,'w');
fprintf(fid,formatspec,interparr);
fclose(fid);