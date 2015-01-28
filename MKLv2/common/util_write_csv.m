function [] = util_write_csv(output_file_name, cell_array)

fid = fopen(output_file_name, 'w') ;

[n_features, dim] = size(cell_array);

for i=1:n_features
    fprintf(fid, '%s', char(cell_array{i, 1}));
    fprintf(fid, ',%f', cell_array{i, 2:end});
    fprintf(fid, '\n');
end

fclose(fid) ;
