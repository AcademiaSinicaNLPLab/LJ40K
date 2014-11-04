
clear all
close all

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

% Path to SimpleMKL package
addpath('/tools/SimpleMKL');

% happy, _happy -> 1, -1
classcode=[1 -1];

nbiter=1;
C = [100];
verbose=1;

feature_root='/home/lwkulab/maxis/projects/LJ40K/exp/train/';

image_feature_name='rgba_gist+rgba_phog';
text_feature_name='TFIDF+keyword';

image_feature_root=[feature_root,image_feature_name,'/csv'];
text_feature_root=[feature_root,text_feature_name,'/csv'];

%% enumerate all emotions
dirInfo = dir(image_feature_root);              % list dir
csv_files = {dirInfo(:).name}';                 % get files
csv_files(ismember(csv_files,{'.','..'})) = []; % remove . and ..

% get all emotions
% emotions = java.util.HashSet;
% for i=1:size(csv_files)
%     fn=char(csv_files(i));
%     segment=regexp(char(fn), '\.', 'split');
%     % segment: 'rgba_gist+rgba_phog'    'K'    'creative'    'dev'    'csv'
%     e=char(segment(3));
%     emotions.add(e);
% end

% for i=1:size(emotions)
    %% emit one emotion
    % emotion=char(emotions(i));

% fn=[features,'.K.', emotion, '.tr.csv'];

% ------------------------------------------------------
%  Load data
% ------------------------------------------------------
% % load K (train)
K_image_tr = csvread(fullfile(image_feature_root,'rgba_gist+rgba_phog.K.sad.tr.csv'));
K_text_tr = csvread(fullfile(text_feature_root,'TFIDF+keyword.K.sad.tr.csv'));

% % build K (train)
% % the size of K_tr: (1440 x 1440 x 2)
K_tr = zeros(size(K_image_tr,1),size(K_image_tr,2),2);
K_tr(:,:,1) = K_image_tr;
K_tr(:,:,2) = K_text_tr;

% % load y (train)
% % the size of y_tr: (1440 x 1)
y_tr = csvread(fullfile(image_feature_root,'rgba_gist+rgba_phog.y.sad.tr.csv'));

% % load K (test)
K_image_te = csvread(fullfile(image_feature_root,'rgba_gist+rgba_phog.K.sad.dev.csv'));
K_text_te = csvread(fullfile(text_feature_root,'TFIDF+keyword.K.sad.dev.csv'));

% % build K (test)
% % the size of K_te: (1440 x 160 x 2)
K_te = zeros(size(K_image_te,1),size(K_image_te,2),2);
K_te(:,:,1) = K_image_te;
K_te(:,:,2) = K_text_te;

% % load y (test)
% % the size of y_te: (1440 x 1)
y_te = csvread(fullfile(image_feature_root,'rgba_gist+rgba_phog.y.sad.dev.csv'));

for i=1: nbiter

    % [Sigma, Alpsup, w0, pos, history, obj, status] = mklsvm(K_tr, y_tr, C, options, verbose);
    [beta, w, b, posw, story(i), obj(i)] = mklsvm(K_tr, y_tr, C, options, verbose);
    
    %% w: (1189 x 1) ??!
    %% Kt: (1440 x 160)

    Kt = K_te(:,:,1)*beta(1) + K_te(:,:,2)*beta(2);

    % the size of ypred should be (1440 x 1)
    ypred = Kt*w+b;

    bc(i) = mean(sign(ypred)==y_te)

end;



