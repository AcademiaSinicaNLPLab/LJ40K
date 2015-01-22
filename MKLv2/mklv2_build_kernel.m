function [K, weight, info_kernel] = mklv2_build_kernel(X_train, X_dev, kernel_type_vec, kernel_option_vec, kernel_var_vec_cell, mkl_options)

%{

Function:
    mklv2_build_kernel

Description:
    build kernel

Input:
    data                : structure
        X_text          : data in text features
        y_text          : labels in text features
        X_image         : data in image features, correspond to x_text
        y_image         : labels in image features, correspond to y_text
    mkl_options         : SimpleMKL options
    kernel_param        : structure
        type_vec        : model type
        option_vec      : model option
        variable_vec    : model variable
    svm_param           : structure
        C               : SVM parameters, only pick the first element

Output:
    y_predict           : a vector that involves the prediction of test samples
    bc                  : evaluation result in %
    time                : training time in seconds
    sigma               : the weights
    alp_sup             : the weigthed lagrangian of the support vectors
    w0                  : the bias
    pos                 : the indices of SV
    history             : history of the weigths
    obj                 : objective value 

%}


[X_image_train, X_image_dev] = normalizemeanstd(X_image_train, X_image_dev);
[weight_image, info_kernel_image] = ...
    UnitTraceNormalization(X_image_train, kernel_type_vec, kernel_option_vec, kernel_var_vec_cell);
K_image = mklkernel(X_image_train, info_kernel_image, weight_image, mkl_options);
disp(sprintf('weight_image is %ld x %ld', size(weight_image, 1), size(weight_image, 2)));
disp(sprintf('info_kernel_image is %ld x %ld', size(info_kernel_image, 1), size(info_kernel_text, 2)));
disp(sprintf('K_image is %ld x %ld x %ld\n', size(K_image, 1), size(K_image, 2), size(K_image,3))); 
