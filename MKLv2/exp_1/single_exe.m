

clear all
close all

dbstop if error;

train_data_root = '/home/doug919/projects/data/MKLv2/2000samples_4/train';
test_data_root = '/home/doug919/projects/data/MKLv2/2000samples_4/test';
train_data_tag = '800p800n_Xy';
test_data_tag = '200p200n_Xy';


% train it separately for each emotion
%features = {'TFIDF', 'keyword', 'image_rgba_gist', 'image_rgba_phog'};

mklv2_exp_1(1, 'E1_800', {'keyword', 'image_rgba_gist', 'image_rgba_phog', 'TFIDF'}, train_data_root, test_data_root, train_data_tag, test_data_tag);
