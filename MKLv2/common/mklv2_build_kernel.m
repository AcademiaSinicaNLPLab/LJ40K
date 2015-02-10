function [K_train, weight, info_kernel, Xnorm_train, Xnorm_test] = mklv2_build_kernel(kernel_param, dim, X_train, X_test, options, feature_start_idx)

if nargin < 6
    feature_start_idx = [];
end

disp('Building training kernel...')
[weight, info_kernel, Xnorm_train, Xnorm_test] = ...
        mklv2_preprocessing(kernel_param, dim, X_train, X_test, feature_start_idx);
K_train = mklkernel(Xnorm_train, info_kernel, weight, options);

disp(sprintf('weight is %ld x %ld', size(weight, 1), size(weight, 2)));
disp(sprintf('info_kernel is %ld x %ld', size(info_kernel, 1), size(info_kernel, 2)));
disp(sprintf('K_train is %ld x %ld x %ld\n', size(K_train, 1), size(K_train, 2), size(K_train,3))); 

