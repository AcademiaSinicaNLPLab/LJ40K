% Example of how to use the mklsvm for  classification
%
%

clear all
close all

nbiter=1;
ratio=0.5;
mat='ionosphere';
C = [100];
verbose=1;

options.algo='svmclass'; % Choice of algorithm in mklsvm can be either
                         % 'svmclass' or 'svmreg'
%------------------------------------------------------
% choosing the stopping criterion
%------------------------------------------------------
options.stopvariation=0; % use variation of weights for stopping criterion 
options.stopKKT=0;       % set to 1 if you use KKTcondition for stopping criterion    
options.stopdualitygap=1; % set to 1 for using duality gap for stopping criterion

%------------------------------------------------------
% choosing the stopping criterion value
%------------------------------------------------------
options.seuildiffsigma=1e-2;        % stopping criterion for weight variation 
options.seuildiffconstraint=0.1;    % stopping criterion for KKT
options.seuildualitygap=0.01;       % stopping criterion for duality gap

%------------------------------------------------------
% Setting some numerical parameters 
%------------------------------------------------------
options.goldensearch_deltmax=1e-1; % initial precision of golden section search
options.numericalprecision=1e-8;   % numerical precision weights below this value
                                   % are set to zero 
options.lambdareg = 1e-8;          % ridge added to kernel matrix 

%------------------------------------------------------
% some algorithms paramaters
%------------------------------------------------------
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
% Modify: MaxisKao @ Sep. 4 2014
options.efficientkernel=0;         % use efficient storage of kernels 


%------------------------------------------------------------------------
%                   Building the kernels parameters
%------------------------------------------------------------------------
kernelt={'gaussian' 'gaussian' 'poly' 'poly' };
kerneloptionvect={[0.5 1 2 5 7 10 12 15 17 20] [0.5 1 2 5 7 10 12 15 17 20] [1 2 3] [1 2 3]};
variablevec={'all' 'single' 'all' 'single'};


classcode=[1 -1];

data = load([mat ]);
% samples: 351
% features: 33
[samples,features]=size(data.x);

% nbtrain: 175
nbtrain=floor(samples*ratio);

rand('state',0);

for i=1: nbiter
    i
    % xapp:  175 x 33
    % yapp:  175 x 1
    % xtest: 176 x 33
    % ytest: 176 x 1
    [xapp, yapp, xtest, ytest, indice] = CreateDataAppTest(data.x, data.y, nbtrain, classcode);

    % normalization
    [xapp, xtest] = normalizemeanstd(xapp, xtest);


    [kernel, kerneloptionvec, variableveccell] = CreateKernelListWithVariable(variablevec,features,kernelt,kerneloptionvect);

    % Weight: 1 x 442
    % InfoKernel: 1x442 struct array with fields:
    %               kernel
    %               kerneloption
    %               variable
    %               Weigth
    [Weight,InfoKernel] = UnitTraceNormalization(xapp, kernel, kerneloptionvec, variableveccell);

    % K : 175 x 175 x 442
    K = mklkernel(xapp, InfoKernel, Weight, options);
    
    % tic
    [beta,w,b,posw,story(i),obj(i)] = mklsvm(K, yapp, C, options, verbose);
    % timelasso(i) = toc

    % Kt: 176 x 62
    Kt = mklkernel(xtest, InfoKernel, Weight, options, xapp(posw,:), beta);

    % ypred = 176 x 1
    ypred = Kt*w+b;
    
    bc(i) = mean(sign(ypred)==ytest)

end;%



