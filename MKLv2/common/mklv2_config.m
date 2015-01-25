% Example of how to use the mklsvm for  classification
%
%


PROJECT_ROOT = '/home/doug919/projects/github_repo/LJ40K/MKLv2';
DATA_ROOT = '/home/doug919/projects/data/MKLv2';
OUTPUT_PATH = '/home/doug919/projects/data/MKLv2/output';
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
svm_param_C = [0.1 1 10 100 300 500 1000];

%------------------------------------------------------------------------
%                               Misc
%------------------------------------------------------------------------
classcode = [1 -1];;
% get emotion vector located at data root folder
emotions = util_read_csv(fullfile(DATA_ROOT, 'emotion.csv'));
