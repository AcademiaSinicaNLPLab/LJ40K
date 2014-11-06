

config();

% load all emotions
emotions = ReadStrCSV(fullfile(PROJECT_ROOT, 'exp/data/emotion.csv'));

% read max(mkl.C) and write to file
fid = fopen('tuned.C.txt','w');
for i=1: size(emotions,2)
    emotion = emotions{i};
    
    fn = fullfile(pwd, 'log', [emotion,'.MKL.mat']);

    mkl = load( fn );
    [M, I] = max(mkl.bc(:));
    fprintf(fid, '%s\t%.2f\t%.2f\n', emotion, mkl.C(I), M);
end
fclose(fid);
