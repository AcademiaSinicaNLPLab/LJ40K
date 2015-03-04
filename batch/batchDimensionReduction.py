# -*- coding: UTF-8 -*-
import sys, os
sys.path.append("../")
import time, logging, argparse
from threading import Thread
from feelit.features import DimensionReduction

def parse_list(astr):
    result = set()
    for part in astr.split(','):
        result.add(int(part))
    return sorted(result)

def get_arguments():
    parser = argparse.ArgumentParser(description='reduce the dimension of a matrix')
    parser.add_argument('algorithm', metavar='algorithm', 
                        help='Currently support truncatedsvd, e.g., truncatedsvd')
    parser.add_argument('training_data_path', metavar='training_data_path', 
                        help='e.g., ../exp/data/from_mongo/keyword.Xy.train.npz')
    parser.add_argument('testing_data_path', metavar='testing_data_path', 
                        help='e.g., ../exp/data/from_mongo/keyword.Xy.test.npz')
    parser.add_argument('dump_dir', metavar='dump_dir', 
                        help='e.g., ../exp/data/from_mongo/')
    parser.add_argument('-n', '--n_components', metavar='N', type=parse_list, default=[300], 
                        help='n_components (DEFAULT: 300). This can be a list expression, e.g., 150,300,500')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, 
                        help='show messages')
    parser.add_argument('-d', '--debug', action='store_true', default=False, 
                        help='show debug messages')
    args = parser.parse_args()
    return args

def DimReduction(i, n_components, algorithm, training_data_path, testing_data_path, dump_dir):
    logging.info("thread %d start! reduce dimension to %d components" % (i,n_components))
    dim_reduction = DimensionReduction(algorithm,logger=logging)

    logging.info("thread %d, %d components, loading training and testing files" % (i,n_components))
    dim_reduction.load_file(training_data_path, testing_data_path)
    dim_reduction.file_sparse_to_dense()

    logging.info('thread %d, %d components, fit' % (i,n_components))
    dim_reduction.file_dimension_reduction_fit(n_components)

    logging.info('thread %d, %d components, transform' % (i,n_components))
    dim_reduction.file_dimension_reduction_transform()

    logging.info('thread %d, %d components, dump files' % (i,n_components))
    dim_reduction.dump(dump_dir)

    logging.info('thread %d, %d components, FINISH' % (i,n_components))

def ArrangeThread(args):
    for i,n in enumerate(args.n_components):
        t = Thread(target=DimReduction, args=(i, n, args.algorithm, args.training_data_path, args.testing_data_path, args.dump_dir))
        time.sleep(10)
        t.start()

if __name__ == '__main__':
    args = get_arguments()

    if args.debug:
        loglevel = logging.DEBUG
    elif args.verbose:
        loglevel = logging.INFO
    else:
        loglevel = logging.ERROR
    logging.basicConfig(format='[%(levelname)s] %(message)s', level=loglevel) 
    
    ArrangeThread(args)
