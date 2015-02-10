function [] = mklv2_make_seed(output_file, npos, nneg)

seed.positive = randperm(npos);;
seed.negative = randperm(nneg);

save(output_file, 'seed')
