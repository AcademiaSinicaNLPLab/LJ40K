
% source_dir
%   e.g., '~/projects/LJ40K/images/data/images/emotion-imgs-1x1-rgb/pattern/'
%
% dest_dir
%   e.g., '~/projects/LJ40K/images/data/test'

function []=batch(source_dir, dest_dir)
    
    % get directory, i.e., emotions, under `source_dir`
    dirInfo = dir(source_dir);                      % list dir
    isub = [dirInfo(:).isdir];                      % returns logical vector
    nameFolds = {dirInfo(isub).name}';                    % get names
    nameFolds(ismember(nameFolds,{'.','..'})) = []; % remove . and ..
    
    for i = 1:size(nameFolds,1)
        emotion = nameFolds{i};
        % fld_path : ...emotion-imgs-1x1-rgb/pattern/accomplished
        fld_path = fullfile(source_dir, emotion);
        
        % collect all filename of all images under `fld_path`        
        fnames = dir(fullfile(fld_path, '*.png'));
        num_files = size(fnames,1);
        filenames = cell(num_files,1);
        for f = 1:num_files
            filenames{f} = fnames(f).name;
        end
        
        % extract features
        disp( ['extracting features from ', fld_path] );
        phog_gen(filenames, fld_path, dest_dir);
end

disp('all done');

