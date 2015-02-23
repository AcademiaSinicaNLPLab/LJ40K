from __future__ import print_function
import sys
import logging
sys.path.append('..')
from feelit.features import PatternFetcher
import numpy as np
import argparse


def get_arguments(argv):

    parser = argparse.ArgumentParser(description='fetch patterns from MongoDB and sum up all vectors')
    parser.add_argument('output_file', metavar='output_file', 
                        help='file name of the ouput .npa file')
    parser.add_argument('-s', '--scoring', action='store_true', default=False, 
                        help='use scored pattern emotion array')
    parser.add_argument('-l', '--vlambda', metavar='LAMBDA', type=float, default=1.0, 
                        help='a scoring parameter lambda which is useful when "-s" is set (DEFAULT: 1.0)')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, 
                        help='show messages')
    parser.add_argument('-d', '--debug', action='store_true', default=False, 
                        help='show debug messages')
    args = parser.parse_args(argv)
    return args

def update_progress_bar(n_cur, n_total, bar_length=50):

    percent = float(n_cur) / n_total
    hashes = '#' * int(round(percent * bar_length))
    spaces = ' ' * (bar_length - len(hashes))
    print('\rPercent: [{0}] {1}%'.format(hashes + spaces, int(round(percent * 100))), end='')


if __name__ == '__main__':

    args = get_arguments(sys.argv[1:])

    if args.debug:
        loglevel = logging.DEBUG
    elif args.verbose:
        loglevel = logging.INFO
    else:
        loglevel = logging.ERROR
    logging.basicConfig(format='[%(levelname)s] %(message)s', level=loglevel) 

    pf = PatternFetcher(logger=logging)


    logging.debug('fetching doc labels')
    # [(udocId0, emotion0), ...], which is sorted by udocId
    docs = pf.get_all_doc_labels()


    logging.info('forming patterns')
    X = []
    y = []
    min_count = 1
    weighted = True    
    for udocId, emotion in docs:

        if loglevel <= logging.INFO:
            update_progress_bar(udocId, len(docs))

        pattern_freq_vec = pf.get_pattern_freq_by_udocId(udocId, min_count, weighted)

        # sum vectors horizontally
        if args.scoring:
            sum_vec = pf.sum_pattern_score_vector(pattern_freq_vec, args.vlambda)
        else:
            sum_vec = pf.sum_pattern_freq_vector(pattern_freq_vec)
        
        X.append(sum_vec)
        y.append(emotion)

    logging.info('save to "%s"' % (args.output_file))
    np.savez_compressed(args.output_file, X=np.array(X), y=np.array(y))
