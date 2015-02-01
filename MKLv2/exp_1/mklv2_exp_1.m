function [] = mklv2_exp_1(emotion_idx, output_prefix, features, train_data_root, test_data_root, train_data_tag, test_data_tag, nclass_neg, seed)


%{

Function:
    mklv2_exp_1

Description:
    (1)-type experiment: TFIDF, keyword, gist, phog

Input:
    emotion_idx             : emotion index in emotion.csv
    output_prefix           : file prefix for output files
    features                : features that are going to train
    train_data_root         : the folder that includes training data
    test_data_root           : the folder that includes testing data
    train_data_tag          : training data tag; e.g. 800p800n_Xy
    test_data_tag           : test data tag; e.g. 800p800n_Xy
    nclass_neg              : number of class of negative samples; e.g. 39
    seed                    : structure; permutation of indexes; if not specified this function would generate a new one
        .positive   : n_train(1) permutatoin of indexes
        .negative   : n_train(2) permutatoin of indexes
    

Output:
    

%}

if nargin < 9
    seed = [];
end

mklv2_exp_1_config;

for i=1:length(features) 

    %load data
    data_file_path = fullfile(train_data_root, features{i}, train_data_tag, sprintf('%s.%s.%s.train.mat', features{i}, train_data_tag, emotions{emotion_idx}));
    [X_orig, y_orig] = mklv2_load_data(data_file_path);

    % separate samples into training and development set
    [n_data, dim] = size(X_orig);
    disp(sprintf('X_orig is %ld x %ld.', n_data, dim));
    n_pos_samples = floor(0.9*n_data/2); 
    n_neg_samples = n_pos_samples;

    [X_train, y_train, X_dev, y_dev, seed] = ...
        mklv2_separate_samples(X_orig, y_orig, [n_pos_samples, n_neg_samples], seed);
    
    %seed_file_name = sprintf('temp_seed_f%ld_e%ld.mat', i, emotion_idx);
    %save(seed_file_name, 'seed');
    disp(sprintf('%ld text samples are separated into %ld training samples and %ld development samples', ...
        n_data, size(y_train, 1), size(y_dev, 1)));


    %------------------------------------------------------------------
    %                           Training and Developing
    %------------------------------------------------------------------
    disp(sprintf('\n===============BUILD KERNEL===================='));
    [K_train, weight, info_kernel, Xnorm_train, Xnorm_dev] = mklv2_build_kernel(kernel_param, dim, X_train, X_dev, options);

    file_prefix = sprintf('%s/%s_%s_%s_%s', OUTPUT_PATH, output_prefix, train_data_tag, emotions{emotion_idx}, features{i});
    kernel_file_path = sprintf('%s_kernels.mat', file_prefix);
    disp(sprintf('<== save to %s', kernel_file_path));
    save(kernel_file_path, 'K_train', 'seed', 'info_kernel', 'weight');


    disp(sprintf('\n===============TRAINING===================='));
    % init result
    result.svm_C = cell(1, length(svm_param_C));
    result.y_predict = cell(1, length(svm_param_C));
    result.bc = cell(1, length(svm_param_C));
    result.time = cell(1, length(svm_param_C));
    result.sigma = cell(1, length(svm_param_C));
    result.alp_sup = cell(1, length(svm_param_C));
    result.w0 = cell(1, length(svm_param_C));
    result.sv_pos = cell(1, length(svm_param_C));
    result.history = cell(1, length(svm_param_C));
    result.obj = cell(1, length(svm_param_C));

    for j=1:length(svm_param_C)        
        result.svm_C{j} = svm_param_C(j);
        [result.y_predict{j}, result.bc{j}, result.time{j}, result.sigma{j},  result.alp_sup{j}, result.w0{j}, result.sv_pos{j}, result.history{j}, result.obj{j}] = ...
            mklv2_train_and_dev(K_train, weight, info_kernel, Xnorm_train, y_train, Xnorm_dev, y_dev, options, svm_param_C(j));
    end

    best_param_C = mklv2_get_best_param(result);
    disp(sprintf('best param C = %ld', best_param_C));

    training_file_path = sprintf('%s_train_result.mat', file_prefix);
    disp(sprintf('<== save to %s', training_file_path));
    save(training_file_path, 'result', 'best_param_C');
    
    clear K_train X_dev X_train Xnorm_dev Xnorm_train y_dev y_train info_kernel weight n_pos_samples n_eng_samples;

    %------------------------------------------------------------------
    %                           Retrain
    %------------------------------------------------------------------
    X_train = X_orig;
    y_train = y_orig;

    % load test data
    %test_data_path = fullfile(test_data_root, features{i}, test_data_tag, sprintf('%s.%s.%s.test.mat', features{i}, test_data_tag, emotions{emotion_idx}));
    test_data_path = fullfile(test_data_root, features{i}, test_data_tag, sprintf('%s.%s.test.mat', features{i}, test_data_tag));
    [X_test, y_test] = mklv2_load_data(test_data_path);

    if (size(X_test, 2) ~= size(X_train, 2))
        error('unmatched dimension');
        keyboard
    end

    
    disp(sprintf('\n===============RE-BUILD KERNEL===================='));
    [K_train, weight, info_kernel, Xnorm_train, Xnorm_test] = mklv2_build_kernel(kernel_param, dim, X_train, X_test, options);

    % save result
    rekernel_file_path = sprintf('%s_rekernel.mat', file_prefix);
    disp(sprintf('<== save to %s', rekernel_file_path));
    save(rekernel_file_path, 'K_train', 'weight', 'info_kernel');

    disp(sprintf('\n===============RE-TRAIN===================='));
    [time, sigma, alp_sup, w0, sv_pos, history, obj] = mklv2_training(kernel_param, options, K_train, y_train, best_param_C);

    % save result
    retrain_file_path = sprintf('%s_retrain.mat', file_prefix);
    disp(sprintf('<== save to %s', retrain_file_path));
    save(retrain_file_path, 'sigma', 'alp_sup', 'w0', 'sv_pos', 'history', 'obj');


    disp(sprintf('\n===============TESTING===================='));
    y_test_binary = mklv2_get_binary_vector(y_test, emotions{emotion_idx});
    [y_predict, bc] = mklv2_testing(Xnorm_train, Xnorm_test, y_test_binary, info_kernel, weight, options, alp_sup, w0, sv_pos, sigma, nclass_neg);
    
    test_file_path = sprintf('%s_test_result.mat', file_prefix);
    disp(sprintf('<== save to %s', test_file_path));
    save(test_file_path, 'y_predict', 'bc');

end
