% source_dir
%   e.g., /home/lwkulab/maxis/projects/LJ40K/images/data/mat/gist/rgb
%
% dest_dir
%   e.g., /home/lwkulab/maxis/projects/LJ40K/images/data/csv/gist/rgb

function []=BatchToCSV(source_dir, dest_dir)
    dirInfo = dir(source_dir);                      % list dir
    
    mat_files = {dirInfo(:).name}';                  % get files
    mat_files(ismember(mat_files,{'.','..'})) = []; % remove . and ..
    
    for i = 1:size(mat_files)
        mat_name = mat_files{i};
        csv_name = strrep(mat_name, '.mat', '.csv');

        mat_path = fullfile(source_dir, mat_name);        
        csv_path = fullfile(dest_dir, csv_name);

        obj = load(mat_path);
        
        if strfind(mat_path, 'gist')
            % disp(['Got gist in ', mat_name]);
            data = obj.gist;
        elseif strfind(mat_path, 'phog')
            % disp(['Got phog in ', mat_name]);
            data = obj.phog;
        else 
            disp(['unknown feature in ', mat_name]);
            continue
        end 
        csvwrite([csv_path], data);
        disp(['convert "', mat_name, '" to "', csv_name, '"']);
    end
