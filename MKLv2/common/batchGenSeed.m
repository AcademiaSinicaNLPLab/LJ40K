

output_file = 'fixed_exp_seed.mat'
n_data = 1600;  % training data
npos = n_data/2; ;
nneg = npos;

mklv2_make_seed(output_file, npos, nneg)

