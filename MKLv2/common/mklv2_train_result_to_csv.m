function [] = mklv2_train_result_to_csv(features, emotion_file_path, input_data_folder, exp_tag, sample_tag, output_file_name)


emotions = util_read_csv(emotion_file_path);
emotion_bc = cell(length(features), length(emotions)+1);
for i=1:length(features)
    emotion_bc{i, 1} = features(i);
    for j=2:length(emotions)+1
        file_name = sprintf('Thread%d_%s_%s_%s_%s_train_result.mat', j-1, exp_tag, sample_tag, emotions{j-1}, features{i});
        data_file_path = fullfile(input_data_folder, file_name);

        maxbc = 0.0;
        if exist(data_file_path, 'file')
            load(data_file_path);
            
            %Name              Size             Bytes  Class     Attributes

            %best_param_C      1x1                  8  double
            %result            1x1             142472  struct

            % get largest bc
            best_idx = 0;
            for k=1:length(result.svm_C)
                if result.svm_C{k}==best_param_C
                    best_idx = k;
                end
            end

            maxbc = result.bc{best_idx};
        else
            warning(sprintf('file "%s" does not exist', data_file_path));
        end
        
        emotion_bc{i, j} = maxbc;
    end
end

util_write_csv(output_file_name, emotion_bc);
