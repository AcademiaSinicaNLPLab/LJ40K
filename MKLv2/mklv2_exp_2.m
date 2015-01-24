
mklv2_config;

%TODO: improve input selection
text_train_sample_dir = '~/projects/data/MKLv2/200samples/train/TFIDF+keyword_eachfromMongo/160_Xy';
image_train_sample_dir = '~/projects/data/MKLv2/200samples/train/rgba_gist+rgba_phog_fromfile/160_Xy';

emotions = {'accomplished'};
features = {'TFIDF+keyword_eachfromMongo', 'rgba_gist+rgba_phog_fromfile'};
filepath_text_data = sprintf('%s/%s.Xy.%s.train.mat', text_train_sample_dir, features{1}, emotions{1});
filepath_image_data = sprintf('%s/%s.Xy.%s.train.mat', image_train_sample_dir, features{2}, emotions{1});

load(filepath_text_data);
data.X_text = X;
data.y_text = y;
load(filepath_image_data);
data.X_image = X;
data.y_image = y;
%data.X_text = [1 2 3; 4 5 6; 7 8 9; 10 11 12; 13 14 15; 16 17 18];
%data.y_text = [1; 1; 1; -1; -1; -1];

%data.X_image = [21 22 23 41; 24 25 26 42; 27 28 29 43; 30 31 32 44; 33 34 35 45; 36 37 38 46];
%data.y_image = [1; 1; 1; -1; -1; -1];

%[y_predict, bc, time, sigma,  Alpsup, w0, pos, history, obj] = mklv2_training(data, options, kernel_param, svm_param);
[y_predict, bc, time, sigma,  Alpsup, w0, pos, history, obj] = mklv2_training(data, options, kernel_param, svm_param_C(1));


keyboard;
