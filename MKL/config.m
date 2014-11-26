
PROJECT_ROOT='/home/lwkulab/maxis/projects/LJ40K';
SimpleMKL_PATH='/tools/SimpleMKL';

LOG_PATH='/home/lwkulab/maxis/projects/LJ40K/MKL/log_full';
EVAL_PATH='/home/lwkulab/maxis/projects/LJ40K/MKL/eval';

if not(exist(LOG_PATH, 'dir'))
    mkdir(LOG_PATH);
end

% `cost` in SVM
C=[0.01, 0.1, 0.5, 1, 2, 4, 10, 50, 100, 150, 200];

% happy, _happy -> 1, -1
classcode=[1 -1];

verbose=1;

% % ====================================================================== % %
% %                               SVM options                              % %
% % ====================================================================== % %

options.algo='svmclass'; % Choice of algorithm in mklsvm can be either
                         % 'svmclass' or 'svmreg'
%------------------------------------------------------
% choosing the stopping criterion
%------------------------------------------------------
options.stopvariation=0;    % use variation of weights for stopping criterion 
options.stopKKT=0;          % set to 1 if you use KKTcondition for stopping criterion
options.stopdualitygap=1;   % set to 1 for using duality gap for stopping criterion

% %------------------------------------------------------
% % choosing the stopping criterion value
% %------------------------------------------------------
options.seuildiffsigma=1e-2;        % stopping criterion for weight variation 
options.seuildiffconstraint=0.1;    % stopping criterion for KKT
options.seuildualitygap=0.01;       % stopping criterion for duality gap

% %------------------------------------------------------
% % Setting some numerical parameters 
% %------------------------------------------------------
options.goldensearch_deltmax=1e-1; % initial precision of golden section search
options.numericalprecision=1e-8;   % numerical precision weights below this value
                                   % are set to zero 
options.lambdareg = 1e-8;          % ridge added to kernel matrix 

% %------------------------------------------------------
% % some algorithms paramaters
% %------------------------------------------------------
options.firstbasevariable='first'; % tie breaking method for choosing the base 
                                   % variable in the reduced gradient method 
options.nbitermax=500;             % maximal number of iteration  
options.seuil=0;                   % forcing to zero weights lower than this 
options.seuilitermax=10;           % value, for iterations lower than this one 

options.miniter=0;                 % minimal number of iterations 
options.verbosesvm=0;              % verbosity of inner svm algorithm 

%
% Note: set 1 would raise the `strrep`
%       error in vectorize.dll
%       and this error is not able to fix
%       because of the missing .h libraay files
% Modify: MaxisKao
options.efficientkernel=0;         % use efficient storage of kernels 
