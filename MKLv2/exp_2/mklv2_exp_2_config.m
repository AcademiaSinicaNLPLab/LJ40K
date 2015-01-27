addpath('/home/doug919/projects/github_repo/LJ40K/MKLv2/common');
mklv2_config;

%------------------------------------------------------------------------
%                   Building the kernels parameters
%------------------------------------------------------------------------
% param for E11 (exhaust memory when >20000 features)
%kernel_param.type_vec = {'gaussian' 'gaussian'};
%kernel_param.option_vec = {[0.1 1 10 15 20] [0.1 1 10 15 20]};
%kernel_param.variable_vec = {'all' 'single'};

% param for E12
%kernel_param.type_vec = {'gaussian' 'gaussian'};
%kernel_param.option_vec = {[0.1 1 10 15 20] [0.1 1 10 15 20]};
%kernel_param.variable_vec = {'all' 'random'};

% param for E13 (total 1*5*10=50 kernels)
kernel_param.type_vec = {'gaussian' 'gaussian' 'gaussian' 'gaussian' 'gaussian' 'gaussian' 'gaussian' 'gaussian' 'gaussian' 'gaussian'};
kernel_param.option_vec = {[0.1 1 10 15 20] [0.1 1 10 15 20] [0.1 1 10 15 20] [0.1 1 10 15 20] [0.1 1 10 15 20] [0.1 1 10 15 20] [0.1 1 10 15 20] [0.1 1 10 15 20] [0.1 1 10 15 20] [0.1 1 10 15 20]};
kernel_param.variable_vec = {'all' 'random' 'random' 'random' 'random' 'random' 'random' 'random' 'random' 'random'};

%------------------------------------------------------------------------
%                   Building the SVM parameters
%------------------------------------------------------------------------
svm_param_C = [10 30 80 100 300 500 1000];
