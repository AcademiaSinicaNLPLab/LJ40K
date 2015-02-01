function [X_return, y_return] =  mklv2_load_data(file_path)

disp(sprintf('==> load from %s', file_path));
load(file_path);
X_return = X;
y_return = y;
