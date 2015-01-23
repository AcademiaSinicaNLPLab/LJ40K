%function [y_predict, bc, time, sigma,  alp_sup, w0, pos, history, obj] = mklv2_training(data, mkl_options, kernel_param, svm_param_C)
function [] = mklv2_training(data, mkl_options, kernel_param, svm_param_C)

%{

Function:
    mklv2_run

Description:
    run the svm training with the specified data file, file_path, and other parameters.

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
    svm_param_C         : a constant; SVM parameter

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

[X_text_nbdata, X_text_dim] = size(data.X_text);
[X_image_nbdata, X_image_dim] = size(data.X_image);

disp(sprintf('X_text is %ld x %ld.', X_text_nbdata, X_text_dim));
disp(sprintf('X_image is %ld x %ld.', X_image_nbdata, X_image_dim));

if X_text_nbdata~=X_image_nbdata
    error('unmatched number of samples in text and image');
end

% the input data would be half positive and half negative
% we extract 90% of each polarity to form the training set
% and 10% to form the testing set
n_pos_samples = floor(0.9*X_text_nbdata/2); %X_image_nbdata is equal to X_text_nbdata
n_neg_samples = n_pos_samples;

rng('shuffle');


% separate samples into training and development set
[X_text_train, y_text_train, X_text_dev, y_text_dev, aux] = ...
    mklv2_separate_samples(data.X_text, data.y_text, [n_pos_samples, n_neg_samples]);
disp(sprintf('%ld text samples are separated into %ld training samples and %ld development samples', ...
    X_text_nbdata, size(y_text_train, 1), size(y_text_dev, 1)));

[X_image_train, y_image_train, X_image_dev, y_image_dev, aux] = ...
    mklv2_separate_samples(data.X_image, data.y_image, [n_pos_samples, n_neg_samples], aux);
disp(sprintf('%ld image samples are separated into %ld training samples and %ld development samples', ...
    X_image_nbdata, size(y_image_train, 1), size(y_image_dev, 1)));

%------------------------------------------------------------------
%                       build text kernels
%------------------------------------------------------------------
disp('Building text kernel...')
[weight_text, info_kernel_text, Xnorm_text_train_norm, Xnorm_text_dev] = ...
    mklv2_normalization(kernel_param, X_text_dim, X_text_train, X_text_dev);
K_text_train = mklkernel(Xnorm_text_train_norm, info_kernel_text, weight_text, mkl_options);

disp(sprintf('weight_text is %ld x %ld', size(weight_text, 1), size(weight_text, 2)));
disp(sprintf('info_kernel_text is %ld x %ld', size(info_kernel_text, 1), size(info_kernel_text, 2)));
disp(sprintf('K_text_train is %ld x %ld x %ld\n', size(K_text_train, 1), size(K_text_train, 2), size(K_text_train,3))); 


%------------------------------------------------------------------
%                       build image kernels
%------------------------------------------------------------------
disp('Building image kernel...')
[weight_image, info_kernel_image, Xnorm_image_train_norm, Xnorm_image_dev] = ...
    mklv2_normalization(kernel_param, X_image_dim, X_image_train, X_image_dev);
K_image_train = mklkernel(Xnorm_image_train_norm, info_kernel_image, weight_image, mkl_options);

disp(sprintf('weight_image is %ld x %ld', size(weight_image, 1), size(weight_image, 2)));
disp(sprintf('info_kernel_image is %ld x %ld', size(info_kernel_image, 1), size(info_kernel_text, 2)));
disp(sprintf('K_image_train is %ld x %ld x %ld\n', size(K_image_train, 1), size(K_image_train, 2), size(K_image_train,3))); 


%------------------------------------------------------------------
%                   pile up two kinds of kernels
%------------------------------------------------------------------
weight_mixed_train = cat(2, weight_text, weight_image);
info_kernel_mixed_train = cat(2, info_kernel_text, info_kernel_image);
K_mixed_train = cat(3, K_text_train, K_image_train);

if ~isequal(y_text_train, y_image_train)    % make sure y is correct
    error('data y error');
end

disp(sprintf('weight_mixed_train is %ld x %ld', size(weight_mixed_train, 1), size(weight_mixed_train, 2)));
disp(sprintf('info_kernel_mixed_train is %ld x %ld', size(info_kernel_mixed_train, 1), size(info_kernel_mixed_train, 2)));
disp(sprintf('K_mixed_train is %ld x %ld x %ld\n', size(K_mixed_train, 1), size(K_mixed_train, 2), size(K_mixed_train,3))); 


%------------------------------------------------------------------
%                           Learning
%------------------------------------------------------------------
disp('Start training...');
verbose = 1;
tic                                                             
% y_text_train is equal to y_image_train
[sigma, alp_sup, w0, pos, history, obj] = mklsvm(K_mixed_train, double(y_text_train), svm_param_C, mkl_options, verbose);
time = toc
disp('Finish training!!!');


%------------------------------------------------------------------
%                     Build Development Kernel
%------------------------------------------------------------------
K_dev = mklv2_make_test_kernel(X_text_dev, X_image_dev, weight_mixed_train, ...
    info_kernel_mixed_train, mkl_options, X_text_train(pos,:), X_image_train(pos,:), sigma);
disp(sprintf('K_dev is %ld x %ld\n', size(K_dev, 1), size(K_dev, 2))); 


%------------------------------------------------------------------
%                     Development Evaluation
%------------------------------------------------------------------
ypred = K_dev * alp_sup + w0;
bc = mean(sign(ypred)==y_text_dev)

keyboard;

