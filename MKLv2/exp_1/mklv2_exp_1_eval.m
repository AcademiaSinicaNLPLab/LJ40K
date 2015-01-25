function [eval_result] = mklv2_exp_1_eval(X_train, Xnorm_train, info_kernel, weight, result, options, test_data_path)

% load test data X, y
disp(sprintf('==> load from %s', test_data_path));
load(test_data_path);
X_test = X;
y_test = y;

% find the best performance index
max_bc = max([result.bc{:}]);
max_idx = find([result.bc{:}] == max_bc)

% make X normalize again
[Xnorm_train_drop, Xnorm_test] = normalizemeanstd(X_train, X_test);

eval_result.idx = cell(1, length(max_idx));
eval_result.y_predict = cell(1, length(max_idx));
eval_result.bc = cell(1, length(max_idx));

% evaluate test data
for i=1:length(max_idx)
    % build test kernel
    disp(sprintf('evaluating the result of index %d ...', max_idx(i)));

    K_test = mklkernel(Xnorm_test, info_kernel, weight, options, Xnorm_train(result.sv_pos{max_idx(i)}, :), result.sigma{max_idx(i)});
    disp(sprintf('K_test is %ld x %ld\n', size(K_test, 1), size(K_test, 2))); 
    
    % display selected param
    disp('show SIGMA:');
    result.sigma{max_idx(i)}

    % eval
    eval_result.idx{i} = max_idx(i);
    eval_result.y_predict{i} = K_test * result.alp_sup{max_idx(i)} + result.w0{max_idx(i)};
    eval_result.bc{i} = mean(sign(eval_result.y_predict{i})==y_test);
end

