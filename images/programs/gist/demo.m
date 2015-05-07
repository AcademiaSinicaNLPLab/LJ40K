% imageFileList: given a set of image files
% imageBaseDir: source path
% dataBaseDir:  output path 

% ex:     fnames = dir(fullfile(fld_path, '*.jpg'));  % get "*.jpg" files
%         num_files = size(fnames,1);                 
%         filenames = cell(num_files,1);             
%    for f = 1:num_files
%        filenames{f} = fnames(f).name;         % put into filenames cell
%    end

% Usage:
%
% imageBaseDir = '~/projects/LJ40K/images/data/images/emotion-imgs-1x1-rgb/pattern/accomplished';
% dataBaseDir = '~/projects/LJ40K/images/data/test';
%
% demo(filenames, imageBaseDir, dataBaseDir);
%

function []=demo(imageFileList, imageBaseDir, dataBaseDir)
    gist=[];
    for f = 1:size(imageFileList,1)
        g=[];
        imageFName = imageFileList{f};
        [dirN fld_name] = fileparts(imageBaseDir);
        outFName = fullfile(dataBaseDir, sprintf('%s_gist.mat', fld_name));

        imageFName = fullfile(imageBaseDir, imageFName);
        img = imread(imageFName);
        
        %% Parameters:
        Nblocks = 4;
        imageSize = size(img,1); 
        orientationsPerScale = [8 8 4];
        numberBlocks = 4;

        %% Precompute filter transfert functions (only need to do this one, unless image size is changes):
        createGabor(orientationsPerScale, imageSize); % this shows the filters
        G = createGabor(orientationsPerScale, imageSize);

        %% Computing gist requires 1) prefilter image, 2) filter image and collect
        %% output energies
        output = prefilt(double(img), 4);
        output=imresize(output, [size(output,1) size(output,1)]);
        g = gistGabor(output, numberBlocks, G);
        if size(g,1)==320
            g=[g;g;g];
        end;
        if size(g,1)<size(gist,1)
            g=gist(:,f-1);
        end;
        gist=[gist g];
        %f
    end;
        sp_make_dir(outFName);
        save(outFName, 'gist');

