function [y_predict, bc, time, sigma,  Alpsup, w0, pos, history, obj] = mklv2_training(data, mkl_options, kernel_param, svm_param)

%{

Function:
    mklv2_run

Description:
    run the svm training with the specified data file, file_path, and other parameters.

Input:
    data                :
        X_text          : data in text features
        y_text          : labels in text features
        X_image         : data in image features, correspond to x_text
        y_image         : labels in image features, correspond to y_text
    mkl_options         : SimpleMKL options
    kernel_param        :
        type_vec        : model type
        option_vec      : model option
        variable_vec    : model variable
    svm_param           : 
        C               : SVM parameters, only pick the first element

Output:
    y_predict           : a vector that involves the prediction of test samples
    bc                  : evaluation result in %
    time                : training time in seconds
    sigma               : the weights
    Alpsup              : the weigthed lagrangian of the support vectors
    w0                  : the bias
    pos                 : the indices of SV
    history             : history of the weigths
    obj                 : objective value 

%}

[n_data, dim] = size(data.X_text);

if [n_data, dim] ~= size(data.X_image)
    error('unmatched dimensions of text and image');
end

disp(sprintf('n_data = %d', n_data));
disp(sprintf('dim = %d', dim));

n_pos_samples = floor(0.9*n_data/2);
n_neg_samples = n_pos_samples;
keyboard%
% build text kernels, 90% for training, 10% for development
[X_text_train, y_text_train, X_text_dev, y_text_dev] = mklv2_separate_samples(data.X_text, data.y_text, [n_pos_samples, n_neg_samples]);


% build text kernels
[X_image_train, y_image_train, X_image_test, y_image_test] = mklv2_separate_samples(data.X_image, data.y_image, [n_pos_samples, n_neg_samples]);



%{

% normalization
[X_training, X_dev] = normalizemeanstd(training_samples, dev_samples);
[kernel, kernel_option_vec, variable_vec_cell] = ...
    CreateKernelListWithVariable(kernel_param.type_vec, dim, kernel_param.kernel_type, kernel_param.option_vec);

% InfoKernel: 
%               kernel
%               kerneloption
%               variable
%               Weigth
[weight, info_kernel] = UnitTraceNormalization(xapp, kernel, kernel_option_vec, variable_vec_cell);

disp(sprintf('kernel is %d x %d', size(kernel, 1), size(kernel, 2)));

K_multiple = 


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


