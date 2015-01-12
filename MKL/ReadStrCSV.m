function [entries] = ReadStrCSV(fn)
    fid = fopen(fn); 
    data = fread(fid, '*char')';
    fclose(fid);
    entries = regexp(strtrim(data), ',', 'split');
