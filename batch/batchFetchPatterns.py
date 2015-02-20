import sys
import logging
sys.path.append('..')
from feelit.patterns import PatternFetcher

if __name__ == '__main__':

    
    loglevel = logging.DEBUG
    logging.basicConfig(format='[%(levelname)s] %(message)s', level=loglevel) 

    pf = PatternFetcher(logger=logging)


    logging.debug('fetching doc labels')
    # [(udocId0, emotion0), ...], which is sorted by udocId
    docs = pf.get_all_doc_labels()


    logging.info('forming patterns')
    min_count = 1
    weighted = True    
    for udocId, emotion in docs:

        pattern_freq_vec = pf.get_pattern_freq_by_udocId(udocId, min_count, weighted)

        # sum vectors horizontally
        import pdb; pdb.set_trace()



