function [X_train, y_train, X_dev, y_dev] = mklv2_kfold(X, y, group_indices, group_id)

% call crossvalind('Kfold', y, 10) to generate variable group_indices
% then use this function to separate train/dev by group_id

dev_bin_idx = (group_indices==group_id);
train_bin_idx = ~dev_bin_idx;

X_dev = X(dev_bin_idx, :);
y_dev = y(dev_bin_idx, :);

X_train = X(train_bin_idx, :);
y_train = y(train_bin_idx, :);



