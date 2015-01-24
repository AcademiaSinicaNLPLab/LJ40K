function [weight, info_kernel, X_normalized_train, X_normalized_dev] = mklv2_normalization(kernel_param, dim, X_train, X_dev)


[kernel_type_vec, kernel_option_vec, kernel_var_vec_cell] = ...
    CreateKernelListWithVariable(kernel_param.variable_vec, dim, kernel_param.type_vec, kernel_param.option_vec);

[X_normalized_train, X_normalized_dev] = normalizemeanstd(X_train, X_dev);
[weight, info_kernel] = ...
    UnitTraceNormalization(X_normalized_train, kernel_type_vec, kernel_option_vec, kernel_var_vec_cell);
