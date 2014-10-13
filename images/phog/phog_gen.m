function []=phog_gen(imageFileList, imageBaseDir, dataBaseDir)
    phog=[];
    % Parameters:
    bin = 8;
    angle = 360;
    L=3;
    roi = [1;256;1;256];
    for f = 1:size(imageFileList,1)
        p=[];
        imageFName = imageFileList{f};
        [dirN fld_name] = fileparts(imageBaseDir);
        outFName = fullfile(dataBaseDir, sprintf('%s_phog.mat', fld_name));

        imageFName = fullfile(imageBaseDir, imageFName);
        I =imageFName;
        p = anna_phog(I,bin,angle,L,roi);

        phog=[phog p];
    end;
        sp_make_dir(outFName);
        save(outFName, 'phog');