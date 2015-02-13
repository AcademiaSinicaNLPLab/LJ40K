addpath('../common');
mklv2_config;

OUTPUT_PATH = '/home/doug919/projects/data/MKLv2/output/exp_4';

%------------------------------------------------------------------------
%                   Building the kernels parameters
%------------------------------------------------------------------------
% param for E4 (total 1*5*4=20 kernels)
kernel_param.type_vec = {'gaussian' 'gaussian' 'gaussian' 'gaussian'};
kernel_param.option_vec = {[0.1 1 10 15 20] [0.1 1 10 15 20] [0.1 1 10 15 20] [0.1 1 10 15 20]};
kernel_param.variable_vec = {'1' '2' '3' '4'};

%------------------------------------------------------------------------
%                   Building the SVM parameters
%------------------------------------------------------------------------
svm_param_C = [10 30 80 100 300 500 1000];
