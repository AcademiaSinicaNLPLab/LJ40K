function [] = mklv2_exp_1(emotion_idx, output_prefix, features)

if nargin < 2
    output_prefix='Exp1'
end

mklv2_exp_1_config;

rng('shuffle');


for i=1:length(features)
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

    %load X, y
    data_file_path = fullfile(DATA_ROOT, '200sample_4/train', features{i}, '160_Xy', sprintf('%s.Xy.%s.train.mat', features{i}, emotions{emotion_idx}));
    disp(sprintf('==> load from %s', data_file_path));
    load(data_file_path);

    % separate samples into training and development set
    [n_data, dim] = size(X);
    disp(sprintf('X is %ld x %ld.', n_data, dim));
    n_pos_samples = floor(0.9*n_data/2); 
    n_neg_samples = n_pos_samples;

    [X_train, y_train, X_dev, y_dev] = ...
        mklv2_separate_samples(X, y, [n_pos_samples, n_neg_samples]);
    disp(sprintf('%ld text samples are separated into %ld training samples and %ld development samples', ...
        n_data, size(y_train, 1), size(y_dev, 1)));

    %------------------------------------------------------------------
    %                       build kernels
    %------------------------------------------------------------------
    disp('Building kernel...')
    [weight, info_kernel, Xnorm_train, Xnorm_dev] = ...
        mklv2_normalization(kernel_param, dim, X_train, X_dev);
    K_train = mklkernel(Xnorm_train, info_kernel, weight, options);

    disp(sprintf('weight is %ld x %ld', size(weight, 1), size(weight, 2)));
    disp(sprintf('info_kernel is %ld x %ld', size(info_kernel, 1), size(info_kernel, 2)));
    disp(sprintf('K_train is %ld x %ld x %ld\n', size(K_train, 1), size(K_train, 2), size(K_train,3))); 


    for j=1:length(svm_param_C)        
        result.svm_C{j} = svm_param_C(j);
        [result.y_predict{j}, result.bc{j}, result.time{j}, result.sigma{j},  result.alp_sup{j}, result.w0{j}, result.sv_pos{j}, result.history{j}, result.obj{j}] = ...
            mklv2_train_one_feature(K_train, weight, info_kernel, Xnorm_train, y_train, Xnorm_dev, y_dev, options, kernel_param, svm_param_C(j));
    end

    file_prefix = sprintf('%s/%s_%s_%s', OUTPUT_PATH, output_prefix, emotions{emotion_idx}, features{i});
    training_file_path = sprintf('%s_train_result.mat', file_prefix);
    disp(sprintf('<== save to %s', training_file_path));
    save(training_file_path, 'result');

    % evaluation
    test_data_path = fullfile(DATA_ROOT, '200sample_4/test', features{i}, '20p20n_Xy', sprintf('%s.20p20n_Xy.%s.test.mat', features{i}, emotions{emotion_idx}));
    eval_result = mklv2_exp_1_eval(X_train, Xnorm_train, info_kernel, weight, result, options, test_data_path)

    eval_file_path = sprintf('%s_eval_result.mat', file_prefix);
    disp(sprintf('<== save to %s', eval_file_path));
    save(eval_file_path, 'eval_result');

end
