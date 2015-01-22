% Example of how to use the mklsvm for  classification
%
%

clear all
close all

addpath('/tools/SimpleMKL');
addpath('/tools/SVM-KM');

options.algo = 'svmclass'; % Choice of algorithm in mklsvm can be either
                         % 'svmclass' or 'svmreg'
%------------------------------------------------------
% choosing the stopping criterion
%------------------------------------------------------
options.stopvariation = 0; % use variation of weights for stopping criterion 
options.stopKKT = 0;       % set to 1 if you use KKTcondition for stopping criterion    
options.stopdualitygap = 1; % set to 1 for using duality gap for stopping criterion

%------------------------------------------------------
% choosing the stopping criterion value
%------------------------------------------------------
options.seuildiffsigma = 1e-2;        % stopping criterion for weight variation 
options.seuildiffconstraint = 0.1;    % stopping criterion for KKT
options.seuildualitygap = 0.01;       % stopping criterion for duality gap

%------------------------------------------------------
% Setting some numerical parameters 
%------------------------------------------------------
options.goldensearch_deltmax = 1e-1; % initial precision of golden section search
options.numericalprecision = 1e-8;   % numerical precision weights below this value
                                   % are set to zero 
options.lambdareg = 1e-8;          % ridge added to kernel matrix 

%------------------------------------------------------
% some algorithms paramaters
%------------------------------------------------------
options.firstbasevariable = 'first'; % tie breaking method for choosing the base 
                                   % variable in the reduced gradient method 
options.nbitermax = 500;             % maximal number of iteration  
options.seuil = 0;                   % forcing to zero weights lower than this 
options.seuilitermax = 10;           % value, for iterations lower than this one 

options.miniter = 0;                 % minimal number of iterations 
options.verbosesvm = 0;              % verbosity of inner svm algorithm 

%
% Note: set 1 would raise the `strrep`
%       error in vectorize.dll
%       and this error is not able to fix
%       because of the missing .h libraay files
% Modify: MaxisKao @ Sep. 4 2014
options.efficientkernel = 0;         % use efficient storage of kernels 


%------------------------------------------------------------------------
%                   Building the kernels parameters
%------------------------------------------------------------------------
kernel_param.type_vec = {'gaussian'};
kernel_param.option_vec = {[0.5 1 5 10 20]};
kernel_param.variable_vec = {'all'};

%------------------------------------------------------------------------
%                   Building the SVM parameters
%------------------------------------------------------------------------
%svm_param.C = [0.1 0.5 1 10 100];
svm_param_C = 1;

%------------------------------------------------------------------------
%                               Misc
%------------------------------------------------------------------------
classcode = [1 -1];;
text_train_sample_dir = '~/projects/data/MKLv2/200samples/train/TFIDF+keyword_eachfromMongo/160_Xy';
image_train_sample_dir = '~/projects/data/MKLv2/200samples/train/rgba_gist+rgba_phog_fromfile/160_Xy';
