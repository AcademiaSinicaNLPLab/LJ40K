function [] = mklv2_exp_4(emotion_idx, output_prefix, features, train_data_root, test_data_root, train_data_tag, test_data_tag, nclass_neg, n_fold)

%{

Function:
    mklv2_exp_4

Description:
    (4)-feature-type experiment: (TFIDF+keyword+gist+phog)

Input:
    emotion_idx             : emotion index in emotion.csv
    output_prefix           : file prefix for output files
    features                : features that are going to train
    train_data_root         : the folder that includes training data
    test_data_root          : the folder that includes testing data
    train_data_tag          : training data tag; e.g. 800p800n_Xy
    test_data_tag           : testing data tag; e.g. 800p800n_Xy
    nclass_neg              : number of class of negative samples; e.g. 39
    n_fold                  : number of folds in cross-validation

Output:
    

%}

mklv2_exp_4_config;


% fuse four features
cells_sample_path = cell(1, length(features));
n_feature_types = length(features);
for i=1:length(features)
    cells_sample_path{i} = fullfile(train_data_root, features{i}, train_data_tag, sprintf('%s.%s.%s.train.mat', features{i}, train_data_tag, emotions{emotion_idx}));
end

feature_string = features{1};
for i=2:n_feature_types
    feature_string = strcat(feature_string, '+');
    feature_string = strcat(feature_string, features{i});
end

[X_fused, y_fused, feature_start_idx] = mklv2_load_multiple_features(cells_sample_path);


[n_data, dim] = size(X_fused);
disp(sprintf('X_fused is %ld x %ld.', n_data, dim));

%if n_fold == 1
    %n_validation = 1;
    %n_pos_samples = floor(0.9*n_data/2); 
    %n_neg_samples = n_pos_samples;

    %[X_train, y_train, X_dev, y_dev, seed] = ...
    %    mklv2_separate_samples(X_fused, y_fused, [n_pos_samples, n_neg_samples], seed);

    %disp(sprintf('%ld text samples are separated into %ld training samples and %ld development samples', ...
    %    n_data, size(y_train, 1), size(y_dev, 1)));
%end


% cross-validation
group_indices = crossvalind('Kfold', y_fused, 10);  % we use 10 instead of n_fold because we want the development set get 10% of training samples when n_fold is 1.
bc_cross_validation = zeros(n_fold, length(svm_param_C));
for i_validation=1:n_fold
    % separate samples into training and development set
    [X_train, y_train, X_dev, y_dev] = mklv2_kfold(X_fused, y_fused, group_indices, i_validation);
    
    disp(sprintf('(%ld) text samples are separated into (%ld) training samples and (%ld) development samples for fold (%d)', ...
         n_data, size(y_train, 1), size(y_dev, 1), i_validation));


    %------------------------------------------------------------------
    %                       build kernels
    %------------------------------------------------------------------
    disp(sprintf('\n===============BUILD KERNEL (fold=%d)====================', i_validation));
    [K_train, weight, info_kernel, Xnorm_train, Xnorm_dev] = mklv2_build_kernel(kernel_param, dim, X_train, X_dev, options, feature_start_idx);

    
    file_prefix = sprintf('%s/%s_%s_%s_%s', OUTPUT_PATH, output_prefix, train_data_tag, emotions{emotion_idx}, feature_string);
    kernel_file_path = sprintf('%s_kernels_fold%d.mat', file_prefix, i_validation);
    disp(sprintf('<== save to %s', kernel_file_path));
    save(kernel_file_path, 'K_train', 'info_kernel', 'weight');


    disp(sprintf('\n===============TRAINING (fold=%d)====================', i_validation));
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

        bc_cross_validation(i_validation, j) = result.bc{j};
    end

    training_file_path = sprintf('%s_train_result_fold%d.mat', file_prefix, i_validation);
    disp(sprintf('<== save to %s', training_file_path));
    save(training_file_path, 'result');

    clear K_train X_dev X_train Xnorm_dev Xnorm_train y_dev y_train info_kernel weight n_pos_samples n_eng_samples;
end

bc_avg = mean(bc_cross_validation);
best_C_idx = find(bc_avg==max(bc_avg));
best_C_idx = best_C_idx(length(best_C_idx));    % we get the last one if we have multiple same value
best_param_C = svm_param_C(best_C_idx);
disp(sprintf('best param C = %ld', best_param_C));


%------------------------------------------------------------------
%                           Retrain
%------------------------------------------------------------------
X_train = X_fused;
y_train = y_fused;

% load testing data
cells_testing_data = cell(1, n_feature_types);
for i=1:n_feature_types
    cells_testing_data{i} = fullfile(test_data_root, features{i}, test_data_tag, sprintf('%s.%s.test.mat', features{i}, test_data_tag));
    %cells_testing_data{i} = fullfile(test_data_root, features{i}, test_data_tag, sprintf('%s.%s.%s.test.mat', features{i}, test_data_tag, emotions{emotion_idx}));
end
[X_test_fused, y_test_fused, feature_test_start_idx] = mklv2_load_multiple_features(cells_testing_data);

if (size(X_test_fused, 2) ~= size(X_train, 2))
    error('unmatched dimension');
    keyboard
end

disp(sprintf('\n===============RE-BUILD KERNEL===================='));
[K_train, weight, info_kernel, Xnorm_train, Xnorm_test] = mklv2_build_kernel(kernel_param, dim, X_train, X_test_fused, options);

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
y_test_binary = mklv2_get_binary_vector(y_test_fused, emotions{emotion_idx});
[y_predict, bc] = mklv2_testing(Xnorm_train, Xnorm_test, y_test_binary, info_kernel, weight, options, alp_sup, w0, sv_pos, sigma, nclass_neg);
    
test_file_path = sprintf('%s_test_result.mat', file_prefix);
disp(sprintf('<== save to %s', test_file_path));
save(test_file_path, 'y_predict', 'bc');

