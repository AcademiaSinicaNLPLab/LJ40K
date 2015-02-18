function [y_predict, bc] = mklv2_testing(Xnorm_train, Xnorm_test, y_test, info_kernel, weight, options, alp_sup, w0, sv_pos, sigma, nclass_neg)

% build testing kernel
K_test = mklkernel(Xnorm_test, info_kernel, weight, options, Xnorm_train(sv_pos, :), sigma);
disp(sprintf('K_test is %ld x %ld\n', size(K_test, 1), size(K_test, 2))); 

y_predict = sign(K_test * alp_sup + w0);

n_test_pos = length(find(y_test==1));
n_test_neg = length(find(y_test==-1));

sum_true_pos = 0;
sum_true_neg = 0;
for i=1:length(y_test)
    if (y_predict(i) == y_test(i)) & (y_test(i) == 1)
        sum_true_pos = sum_true_pos + 1;
    elseif (y_predict(i) == y_test(i)) & (y_test(i) == -1)
        sum_true_neg = sum_true_neg + 1;
    end
end

bc = (sum_true_pos + sum_true_neg / nclass_neg) / (n_test_pos + n_test_neg / nclass_neg)

