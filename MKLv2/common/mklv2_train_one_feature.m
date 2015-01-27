function [y_predict, bc, time, sigma,  alp_sup, w0, sv_pos, history, obj] = mklv2_train_one_feature(K_train, weight, info_kernel, Xnorm_train, y_train, Xnorm_dev, y_dev, options, svm_C)


%------------------------------------------------------------------
%                           Learning
%------------------------------------------------------------------
disp(sprintf('Start training, C=%d...', svm_C));
verbose = 1;
tic;                                                             
% y_text_train is equal to y_image_train
[sigma, alp_sup, w0, sv_pos, history, obj] = mklsvm(K_train, y_train, svm_C, options, verbose);
time = toc;
disp('Finish training!!!');


%------------------------------------------------------------------
%                     Build Development Kernel
%------------------------------------------------------------------
K_dev = mklkernel(Xnorm_dev, info_kernel, weight, options, Xnorm_train(sv_pos, :), sigma);
disp(sprintf('K_dev is %ld x %ld\n', size(K_dev, 1), size(K_dev, 2))); 


%------------------------------------------------------------------
%                     Development Evaluation
%------------------------------------------------------------------
y_predict = K_dev * alp_sup + w0;
bc = mean(sign(y_predict)==y_dev)

