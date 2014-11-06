% Parameters
% ==========
% eid: int
%   the index of sorted emotions (starts with 1)
%   
% Example
% =======
% mkl(1)
%
% 
function [] = mklTrain(eid)

    % change config in config.m
    config();

    % Path to SimpleMKL package
    addpath(SimpleMKL_PATH);
    addpath(fullfile(PROJECT_ROOT,'MKL'));

    % ------------------------------------------------------
    %  Set Pathes
    % ------------------------------------------------------

    feature_root=fullfile(PROJECT_ROOT,'exp/train');

    image_feature_name = 'rgba_gist+rgba_phog';
    text_feature_name = 'TFIDF+keyword';

    image_feature_root = fullfile(feature_root,image_feature_name,'csv');
    text_feature_root = fullfile(feature_root,text_feature_name,'csv');


    % ------------------------------------------------------
    %  Generate Pathes
    % ------------------------------------------------------

    % Read all emotions
    disp('> load emotions');
    emotions = ReadStrCSV(fullfile(PROJECT_ROOT, 'exp/data/emotion.csv'));

    % assembly save path
    % exit if the final file exists
    save_fn = sprintf('%s.MKL.mat',emotions{eid});
    save_path = fullfile(LOG_PATH, save_fn);
    if exist(save_path)
        disp(['file ', save_path, ' exists'])
        exit
    end

    % Set features
    features = {image_feature_name, text_feature_name};

    % rgba_gist+rgba_phog.K.sad.tr.csv
    K_image_tr_fn = sprintf('%s.K.%s.tr.csv', features{1}, emotions{eid});
    % TFIDF+keyword.K.sad.tr.csv
    K_text_tr_fn = sprintf('%s.K.%s.tr.csv', features{2}, emotions{eid});

    % 'rgba_gist+rgba_phog.y.sad.tr.csv'
    y_tr_fn = sprintf('%s.y.%s.tr.csv', features{1}, emotions{eid});

    % 'rgba_gist+rgba_phog.K.sad.dev.csv'
    K_image_te_fn = sprintf('%s.K.%s.dev.csv', features{1}, emotions{eid});
    % 'TFIDF+keyword.K.sad.dev.csv'
    K_text_te_fn = sprintf('%s.K.%s.dev.csv', features{2}, emotions{eid});

    % 'rgba_gist+rgba_phog.y.sad.dev.csv'
    y_te_fn = sprintf('%s.y.%s.dev.csv', features{1}, emotions{eid});

    % ------------------------------------------------------
    %  Load data
    % ------------------------------------------------------
    % % load K (train)
    disp('> load K_image_tr');
    K_image_tr = csvread(fullfile(image_feature_root, K_image_tr_fn));

    disp('> load K_text_tr');
    K_text_tr = csvread(fullfile(text_feature_root, K_text_tr_fn));

    % % build K (train)
    % % the size of K_tr: (1440 x 1440 x 2)
    disp('> build K_tr');
    K_tr = zeros(size(K_image_tr,1),size(K_image_tr,2),2);
    K_tr(:,:,1) = K_image_tr;
    K_tr(:,:,2) = K_text_tr;

    % % load y (train)
    % % the size of y_tr: (1440 x 1)
    disp('> load y_tr');
    y_tr = csvread(fullfile(image_feature_root, y_tr_fn));

    % % load K (test)
    disp('> load K_image_te');
    K_image_te = csvread(fullfile(image_feature_root, K_image_te_fn));
    disp('> load K_text_te');
    K_text_te = csvread(fullfile(text_feature_root, K_text_te_fn));

    % % build K (test)
    % % the size of K_te: (1440 x 160 x 2)
    disp('> build K_te');
    K_te = zeros(size(K_image_te,1),size(K_image_te,2),2);
    K_te(:,:,1) = K_image_te;
    K_te(:,:,2) = K_text_te;

    % % load y (test)
    % % the size of y_te: (1440 x 1)
    disp('> load y_te');
    y_te = csvread(fullfile(image_feature_root, y_te_fn));

    % ------------------------------------------------------
    %  Learning
    % ------------------------------------------------------
    ypred = cell(1, size(C,2));

    disp('> start learning...');

    for i=1: size(C,2)

        % training
        % [Sigma, Alpsup, w0, pos, history, obj, status] = mklsvm(K_tr, y_tr, C, options, verbose);
        tic;
        [beta, w, b, posw, story(i), obj(i)] = mklsvm(K_tr, y_tr, C(i), options, verbose);
        time(i) = toc;

        % compute Kt (160 x 1189)
        Kt = K_te(:,:,1)*beta(1) + K_te(:,:,2)*beta(2);
        Kt = Kt(posw,:)';

        % predict

        ypred{i} = Kt*w+b;

        % evaluation
        bc(i) = mean(sign(ypred{i})==y_te)

    end;

    
    % log `C`, `ypred` and `bc`
    disp(['> saving to ', save_path]);
    save(save_path, 'C', 'ypred', 'bc', 'time');



