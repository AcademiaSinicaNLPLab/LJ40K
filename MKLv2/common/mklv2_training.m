function [time, sigma, alp_sup, w0, sv_pos, history, obj] = ...
    mklv2_training(kernel_param, options, K_train, y_train, svm_C)

disp(sprintf('Start training, C=%d...', svm_C));
tic;                                                             
% y_text_train is equal to y_image_train
[sigma, alp_sup, w0, sv_pos, history, obj] = mklsvm(K_train, y_train, svm_C, options, 1);
time = toc;
disp('Finish retraining!!!');



