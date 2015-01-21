function [X_train, y_train, X_test, y_test] = mklv2_separate_samples(X, y, n_train, classcode)

%{

Function:
    mklv2_separate_samples

Description:
    separate samples into training and testing sets

Input:
    X          : data
    y          : labels
    n_train    : must have two elements, number of {positive, negtive} samples in training set
    classcode  : label representations; default value is [1 -1]

Output:
    X_train    : data of training samples
    y_train    : labels of training samples
    X_test                : training time in seconds
    y_test               : the weights

%}
    
if nargin<4
    classcode(1) = 1;
    classcode(2) = -1;
end;

if length(n_train)~=2
    error('parameter n_train must have two elements.');
else
keyboard 
    n_train_pos = n_train(1);
    n_train_neg = n_train(2);
    idx_pos = find(y==classcode(1));
    idx_neg = find(y==classcode(2));
    n_pos = length(idx_pos);
    n_neg = length(idx_neg);
    aux_pos = randperm(n_pos);
    aux_neg = randperm(n_neg);
    idx_train = [idx_pos(aux_pos(1:n_train_pos)); idx_neg(aux_neg(1:n_train_neg))];
    idx_test = [idx_pos(aux_pos(n_train_pos+1:end)) ; idx_neg(aux_neg(n_train_neg+1:end))];
    X_train = X(idx_train,:);
    y_train = y(idx_train);
    X_test = X(idx_test,:);
    y_test = y(idx_test,:);
end


