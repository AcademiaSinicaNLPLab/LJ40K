function [] = mklv2_eval_result_to_csv(features, emotion_file_path, input_data_folder, exp_tag, sample_tag, output_file_name)

%exp_tag = 'E1_800';
%sample_tag = '800p800n_Xy';
%out_file_name = 'exp_2_eval_result.csv'

%DATA_DIR = '/home/doug919/projects/data/MKLv2';
%emotion_file_path = fullfile(DATA_DIR, 'emotion.csv');
%data_path_prefix = fullfile(DATA_DIR, 'output');
%features = {'image_rgba_gist', 'image_rgba_phog', 'keyword'};

emotions = util_read_csv(emotion_file_path);
emotion_bc = cell(length(features), length(emotions)+1);
for i=1:length(features)
    emotion_bc{i, 1} = features(i);
    for j=2:length(emotions)+1
        file_name = sprintf('Thread%d_%s_%s_%s_%s_eval_result.mat', j-1, exp_tag, sample_tag, emotions{j-1}, features{i});
        data_file_path = fullfile(input_data_folder, file_name);

        maxbc = 0.0;
        if exist(data_file_path, 'file')
            load(data_file_path);
            
            % get largest bc
            for k=1:length(eval_result.bc)
                if maxbc < eval_result.bc{k}
                    maxbc = eval_result.bc{k};
                end
            end
        else
            warning(sprintf('file "%s" does not exist', data_file_path));
        end
        
        emotion_bc{i, j} = maxbc;
    end
end

util_write_csv(output_file_name, emotion_bc);

keyboard
