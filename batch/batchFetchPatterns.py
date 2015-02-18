import sys
import logging
sys.path.append('..')
from feelit.patterns import PatternFetcher

if __name == '__main__':

	
    loglevel = logging.DEBUG
    logging.basicConfig(format='[%(levelname)s] %(message)s', level=loglevel) 

	pf = PatternFetcher(logger=logging)


	logging.debug('fetching doc labels')
	# [(udocId0, emotion0), ...], which is sorted by udocId
	docs = pf.get_all_doc_labels()


	logging.info('forming patterns')
    for udocId, emotion in docs:
    	pf.get_patterns_by_udocId(udocId)


