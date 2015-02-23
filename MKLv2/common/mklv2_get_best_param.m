function [best_param_C] = mklv2_get_best_param(result)

% find the best performance index
max_bc = max([result.bc{:}]);
max_idx = find([result.bc{:}] == max_bc);

% if there are multiple instances I choose the largest one, since larger C usually performs better.
best_param_C = result.svm_C{max_idx(length(max_idx))};
