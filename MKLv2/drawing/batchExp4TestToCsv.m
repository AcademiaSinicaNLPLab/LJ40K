
clear all
close all

dbstop if error;

addpath('../exp_4');
mklv2_exp_4_config;

exp_tag = 'E4_8000';
sample_tag = '800p800n_Xy';
output_file_path = 'output/exp_4_test_result_15020215.csv';

DATA_DIR = '/home/doug919/projects/data/MKLv2';
emotion_file_path = fullfile(DATA_DIR, 'emotion.csv');
input_data_folder = fullfile(DATA_DIR, 'output', 'exp_4');
features = {'TFIDF+keyword+image_rgba_gist+image_rgba_phog'};
file_prefix = 'Seq';

mklv2_test_result_to_csv(features, emotion_file_path, input_data_folder, exp_tag, sample_tag, output_file_path, file_prefix);