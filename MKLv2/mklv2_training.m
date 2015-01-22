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

[n_data, dim] = size(data.X_text);

if [n_data, dim] ~= size(data.X_image)
    error('unmatched dimensions of text and image');
end

disp(sprintf('n_data = %ld', n_data));
disp(sprintf('dim = %ld', dim));

% the input data would be half positive and half negative
% we extract 90% of each polarity to form the training set
% and 10% to form the testing set
n_pos_samples = floor(0.9*n_data/2);
n_neg_samples = n_pos_samples;

rng('shuffle');

% generate kernel option for both text and image
[kernel_type_vec, kernel_option_vec, kernel_var_vec_cell] = ...
    CreateKernelListWithVariable(kernel_param.variable_vec, dim, kernel_param.type_vec, kernel_param.option_vec);

% separate samples into training and development set
[X_text_train, y_text_train, X_text_dev, y_text_dev, aux] = ...
    mklv2_separate_samples(data.X_text, data.y_text, [n_pos_samples, n_neg_samples]);
disp(sprintf('%ld text samples are separated into %ld training samples and %ld development samples', ...
    n_data, size(y_text_train, 1), size(y_text_dev, 1)));

[X_image_train, y_image_train, X_image_dev, y_image_dev, aux] = ...
    mklv2_separate_samples(data.X_image, data.y_image, [n_pos_samples, n_neg_samples], aux);
disp(sprintf('%ld image samples are separated into %ld training samples and %ld development samples', ...
    n_data, size(y_image_train, 1), size(y_image_dev, 1)));


%------------------------------------------------------------------
%                       build text kernels
%------------------------------------------------------------------
[X_text_train, X_text_dev] = normalizemeanstd(X_text_train, X_text_dev);
[weight_text, info_kernel_text] = ...
    UnitTraceNormalization(X_text_train, kernel_type_vec, kernel_option_vec, kernel_var_vec_cell);
K_text_train = mklkernel(X_text_train, info_kernel_text, weight_text, mkl_options);
disp(sprintf('weight_text is %ld x %ld', size(weight_text, 1), size(weight_text, 2)));
disp(sprintf('info_kernel_text is %ld x %ld', size(info_kernel_text, 1), size(info_kernel_text, 2)));
disp(sprintf('K_text_train is %ld x %ld x %ld\n', size(K_text_train, 1), size(K_text_train, 2), size(K_text_train,3))); 


%------------------------------------------------------------------
%                       build image kernels
%------------------------------------------------------------------
[X_image_train, X_image_dev] = normalizemeanstd(X_image_train, X_image_dev);
[weight_image, info_kernel_image] = ...
    UnitTraceNormalization(X_image_train, kernel_type_vec, kernel_option_vec, kernel_var_vec_cell);
K_image_train = mklkernel(X_image_train, info_kernel_image, weight_image, mkl_options);
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
verbose = 1;
tic                                                             
% y_text_train is equal to y_image_train
[sigma, alp_sup, w0, pos, history, obj] = mklsvm(K_mixed_train, y_text_train, svm_param_C, mkl_options, verbose);
time = toc


%------------------------------------------------------------------
%                     Build Development Kernel
%------------------------------------------------------------------
keyboard
%{
K_dev = mklv2_make_test_kernel(X_text_dev, X_image_dev, weight_mixed_train, ...
    info_kernel_mixed_train, mkl_options, X_text_train(pos,:), X_image_train(pos,:), sigma);

sep_idx = size(K_text_train, 3);
K_text_dev = mklkernel(X_text_dev, info_kernel_text, weight_text, mkl_options, X_text_train(pos,:), sigma(1:sep_idx));
K_image_dev = mklkernel(X_image_dev, info_kernel_image, weight_image, mkl_options, X_image_train(pos,:), sigma(sep_idx+1:end));
K_mixed_dev = cat(3, K_text_dev, K_image_dev);
disp(sprintf('K_text_dev is %ld x %ld x %ld\n', size(K_text_dev, 1), size(K_text_dev, 2), size(K_text_dev,3))); 
disp(sprintf('K_image_dev is %ld x %ld x %ld\n', size(K_image_dev, 1), size(K_image_dev, 2), size(K_image_dev,3))); 
disp(sprintf('K_mixed_dev is %ld x %ld x %ld\n', size(K_mixed_dev, 1), size(K_mixed_dev, 2), size(K_mixed_dev,3))); 
%}

%------------------------------------------------------------------
%                     Development Evaluation
%------------------------------------------------------------------


%{

for i=1: nbiter
    i

    % xapp:  175 x 33
    % yapp:  175 x 1
    % xtest: 176 x 33
    % ytest: 176 x 1
    [xapp, yapp, xtest, ytest, indice] = CreateDataAppTest(x, y, nbtrain,classcode);

    disp(sprintf('xapp is %d x %d', size(xapp, 1), size(xapp, 2)));
    disp(sprintf('yapp is %d x %d', size(yapp, 1), size(yapp, 2)));
    disp(sprintf('xtest is %d x %d', size(xtest, 1), size(xtest, 2)));
    disp(sprintf('ytest is %d x %d', size(ytest, 1), size(ytest, 2)));

    % normalization
    [xapp, xtest] = normalizemeanstd(xapp, xtest);


    %[kernel, kerneloptionvec, variableveccell] = CreateKernelListWithVariable(variablevec,dim,kernelt,kerneloptionvect);
    [kernel, kerneloptionvec, variableveccell] = mymkl_CreateKernelListWithVariable(variablevec,dim,kernelt,kerneloptionvect);

    disp(sprintf('kernel is %d x %d', size(kernel, 1), size(kernel, 2)));   

    % Weight: 1 x 442
    % InfoKernel: 1x442 struct array with fields:
    %               kernel
    %               kerneloption
    %               variable
    %               Weigth
    %[Weight,InfoKernel] = UnitTraceNormalization(xapp, kernel, kerneloptionvec, variableveccell);
    [Weight,InfoKernel] = mymkl_UnitTraceNormalization(xapp, kernel, kerneloptionvec, variableveccell);

    % K : 175 x 175 x 442
    K = mklkernel(xapp, InfoKernel, Weight, options);

    disp(sprintf('K is %d x %d x %d', size(K, 1), size(K, 2), size(K,3))); 

    
    %------------------------------------------------------------------
    % 
    %  K is a 3-D matrix, where K(:,:,i)= i-th Gram matrix 
    %
    %------------------------------------------------------------------
    % or K can be a structure with uses a more efficient way of storing
    % the gram matrices
    %
    % K = build_efficientK(K);
    
    tic
    [beta,w,b,posw,story(i),obj(i)] = mklsvm(K, yapp, C, options, verbose);
    timelasso(i) = toc

    % Kt: 176 x 62
    Kt = mklkernel(xtest, InfoKernel, Weight, options, xapp(posw,:), beta);
    disp(sprintf('Kt is %d x %d', size(Kt, 1), size(Kt, 2))); 

    % ypred = 176 x 1
    ypred = Kt*w+b;

    bc(i) = mean(sign(ypred)==ytest)

end;%

%}


