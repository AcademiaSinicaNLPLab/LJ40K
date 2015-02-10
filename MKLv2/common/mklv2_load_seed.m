

seed_file_name = 'fixed_exp_seed.mat';

if exist(seed_file_name)
    disp('use last seed');
    load(seed_file_name);
else
    seed = [];
end

