% Parameters
% ==========
% eid: int
%   the index of sorted emotions (starts with 1)
%   
% Example
% =======
% mklTest(1)
%
% 
function [] = mklTest(eid)

    % change config in config.m
    config();

    % Path to SimpleMKL package
    addpath(SimpleMKL_PATH);
    addpath(fullfile(PROJECT_ROOT,'MKL'));

    % ------------------------------------------------------
    %  Set Pathes
    % ------------------------------------------------------

    feature_root = fullfile(PROJECT_ROOT,'exp/test');

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

    % Load log to obtain best parameter
    
    log_fn = sprintf('%s.MKL.mat',emotions{eid});
    log_path = fullfile(LOG_PATH, log_fn);
    log_data = load(log_path);
    % example of the `log_data` (accomplished.MKL.mat)
    %     C: [0.0100 0.1000 0.5000 1 2 4 10 50 100 150 200]
    % ypred: {[160x1 double]  [160x1 double]  [160x1 double]  [160x1 double]  [160x1 double]  [160x1 double]  [160x1 double]  [160x1 double]  [160x1 double]  [160x1 double]  [160x1 double]}
    % betas: {[1 0]  [1 0]  [1 0]  [1 0]  [1 0]  [1 0]  [1 0]  [0.9985 0.0015]  [0.7977 0.2023]  [0.7929 0.2071]  [0.7929 0.2071]}
    % posws: {[1440x1 double]  [1432x1 double]  [1316x1 double]  [1261x1 double]  [1243x1 double]  [1239x1 double]  [1247x1 double]  [1249x1 double]  [1252x1 double]  [1250x1 double]  [1250x1 double]}
    %    ws: {[1440x1 double]  [1432x1 double]  [1316x1 double]  [1261x1 double]  [1243x1 double]  [1239x1 double]  [1247x1 double]  [1249x1 double]  [1252x1 double]  [1250x1 double]  [1250x1 double]}
    % story: [1x11 struct]
    %   obj: [14.2197 127.9259 450.2751 677.1079 895.4283 1.0397e+03 1.1626e+03 1.5088e+03 1.5809e+03 1.5822e+03 1.5822e+03]
    %  time: [2.3530 1.7432 2.9869 5.3325 10.3696 16.1362 16.0087 34.9603 21.2979 20.3102 20.7756]
    %    bs: {[0.8176]  [-0.4017]  [-0.6723]  [-0.5704]  [-0.5241]  [-0.5609]  [-0.5622]  [-0.6332]  [-0.5732]  [-0.5381]  [-0.5381]}
    %    bc: [0.4938 0.5813 0.6875 0.7000 0.7000 0.6813 0.6813 0.6625 0.6625 0.6687 0.6687]    
    idx = find(log_data.bc==max(log_data.bc));
    C = log_data.C(idx);

    C_indexes = find(log_data.bc==max(log_data.bc));
    min_C_times = log_data.time(C_indexes);

    % get the best index by targeting the best C with minimun training time
    best_idx = C_indexes(find(min_C_times==min(min_C_times)));
    C = log_data.C(best_idx);

    % get the beta, posw with best setting
    beta = log_data.betas{best_idx};
    posw = log_data.posws{best_idx};
    b = log_data.bs{best_idx};
    w = log_data.ws{best_idx};

    % Set features
    features = {image_feature_name, text_feature_name};

    % 'rgba_gist+rgba_phog.K.sad.dev.csv'
    K_image_te_fn = sprintf('%s.K.%s.te.csv', features{1}, emotions{eid});
    % 'TFIDF+keyword.K.sad.dev.csv'
    K_text_te_fn = sprintf('%s.K.%s.te.csv', features{2}, emotions{eid});

    % 'rgba_gist+rgba_phog.y.sad.dev.csv'
    y_te_fn = sprintf('%s.y.%s.te.csv', features{1}, emotions{eid});

    % ------------------------------------------------------
    %  Load data
    % ------------------------------------------------------

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
    %  Predicting
    % ------------------------------------------------------

    disp('> start predicting...');

    % compute Kt (160 x 1189)
    Kt = K_te(:,:,1)*beta(1) + K_te(:,:,2)*beta(2);
    Kt = Kt(posw,:)';

    % predict
    ypred = Kt*w+b;

    % evaluation
    bc = mean(sign(ypred)==y_te);


    eval_fn = sprintf('%s.eval.mat',emotions{eid});
    eval_save_path = fullfile(EVAL_PATH, eval_fn);

    % log `C`, `ypred` and `bc`
    disp(['> saving to ', eval_save_path]);
    save(eval_save_path, 'bc', 'ypred', 'C', 'beta', 'posw', 'b', 'w');



