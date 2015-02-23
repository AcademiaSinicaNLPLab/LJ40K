function [X_fused, y_fused, start_idx] = mklv2_load_multiple_features(cells_sample_path)

%{
    cells_sample_path: cells of string; e.g. {'/home/doug919/a.mat' '/home/doug919/b.mat'}
%}

X_fused = [];
y_fused = [];
start_idx = [];
next_start_idx = 1;

for i=1:length(cells_sample_path)
    disp(sprintf('==> loading data from %s', cells_sample_path{i}));
    load(cells_sample_path{i});
    size_of_X = size(X)
    
    if size(y_fused) == [0 0]
        y_fused = y;
    elseif ~isequal(y_fused, y)
        error('y is not matched');
    end

    X_fused = cat(2, X_fused, X);
    start_idx = cat(2, start_idx, next_start_idx);
    next_start_idx = size(X_fused, 2)+1;
end

