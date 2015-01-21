
mklv2_config;

data.X_text = [1 2 3; 4 5 6; 7 8 9; 10 11 12; 13 14 15; 16 17 18];
data.y_text = [1; 1; 1; -1; -1; -1];

data.X_image = [21 22 23; 24 25 26; 27 28 29; 30 31 32; 33 34 35; 36 37 38];
data.y_image = [1; 1; 1; -1; -1; -1];

[y_predict, bc, time, sigma,  Alpsup, w0, pos, history, obj] = mklv2_training(data, options, kernel_param, svm_param);
