% Example of how to use the mklsvm for  classification
%
%

clear all
close all

PROJECT_ROOT = '/home/doug919/projects/github_repo/LJ40K/MKLv2';
DATA_ROOT = '/home/doug919/projects/data/MKLv2';
OUTPUT_PATH = '/home/doug919/projects/data/MKLv2/output';
addpath('/tools/SimpleMKL');
addpath('/tools/SVM-KM');


nbiter=1;
ratio=0.9;
C = [100];
verbose=1;
features = {'keyword'};
emotions = {'accomplished'};
data_file_path = fullfile(DATA_ROOT, '200sample_4/train', features{1}, '160_Xy', sprintf('%s.Xy.%s.train.mat', features{1}, emotions{1}));
disp(sprintf('==> load from %s', data_file_path));
load(data_file_path);

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
options.efficientkernel=0;         % use efficient storage of kernels 


%------------------------------------------------------------------------
%                   Building the kernels parameters
%------------------------------------------------------------------------
%kernelt={'gaussian' 'gaussian'};
%kerneloptionvect={[0.1 1 10 15 20] [0.1 1 10 15 20]};
%variablevec={'all' 'single'};
kernelt={'gaussian'};
kerneloptionvect={[0.1 1 10 15 20]};
variablevec={'all'};

classcode=[1 -1];
[nbdata,dim]=size(X);
size(X)
nbtrain=floor(nbdata*ratio);
rand('state',0);

for i=1: nbiter
    i
    [xapp,yapp,xtest,ytest,indice]=CreateDataAppTest(X, y, nbtrain,classcode);
    [xapp,xtest]=normalizemeanstd(xapp,xtest);
    [kernel,kerneloptionvec,variableveccell]=CreateKernelListWithVariable(variablevec,dim,kernelt,kerneloptionvect);
    [Weight,InfoKernel]=UnitTraceNormalization(xapp,kernel,kerneloptionvec,variableveccell);
    size(Weight)
    size(InfoKernel)
    K=mklkernel(xapp,InfoKernel,Weight,options);

    size(K)
    
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
    [beta,w,b,posw,story(i),obj(i)] = mklsvm(K,yapp,C,options,verbose);
    timelasso(i)=toc

    Kt=mklkernel(xtest,InfoKernel,Weight,options,xapp(posw,:),beta);
    ypred=Kt*w+b;

    bc(i)=mean(sign(ypred)==ytest)
    keyboard
end;%



