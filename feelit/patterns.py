import logging, os, sys
from feelit import utils
import numpy as np
import pymongo

"""
    See batchFetchPatterns.py for example usage
"""

class PatternFetcher(object):

    def __init__(self, **kwargs):
        """
        options:
            logger          : logging instance
            mongo_addr      : mongo db import                           (DEFAULT: 'doraemon.iis.sinica.edu.tw')
            db              : database name                             (DEFAULT: 'LJ40K')
            lexicon         : pattern frequency collection              (DEFAULT: 'lexicon.nested')
            pats            : patterns related to all the documents     (DEFAULT: 'pats')
            docs            : map of udocId and emotions                (DEFAULT: 'docs')
        """

        ## process args
        if 'logger' in kwargs and kwargs['logger']:
            self.logging = kwargs['logger']
        else:
            logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.ERROR)  
            self.logging = logging

        ## mongodb settings
        mongo_addr = 'doraemon.iis.sinica.edu.tw' if 'mongo_addr' not in kwargs else kwargs['mongo_addr']

        ## default collection name
        self.db = 'LJ40K' if 'db' not in kwargs else kwargs['db']

        lexicon = 'lexicon.nested' if 'lexicon' not in kwargs else kwargs['lexicon']
        pats = 'pats' if 'pats' not in kwargs else kwargs['pats']
        docs = 'docs' if 'docs' not in kwargs else kwargs['docs']

        ### connect to mongodb
        self.mongo_client = pymongo.MongoClient(mongo_addr)

        self.collection_pattern_freq = self.mongo_client[self.db][lexicon]
        self.collection_patterns = self.mongo_client[self.db][pats]
        self.collection_docs = self.mongo_client[self.db][docs]

        color_order = self.mongo_client['feelit']['color.order']
        self.emotion_list = color_order.find_one({ 'order': 'group-maxis'})['emotion']

    def get_all_doc_labels(self, sort=True):
        """
        parameters:
            sort: True/False; sorting by docId
        return:
            [(udocId0, emotion0), ...], which is sorted by udocId
        """
        docs = [(doc['udocID'], doc['emotion']) for doc in self.collection_docs.find().batch_size(1024)]

        if sort:
            docs = sorted(docs, key=lambda x:x[0] )
        return docs

    def get_pattern_freq_by_udocId(self, udocId, min_count=1):

        """
        parameters:
            udocId: the id you want 
            min_count: the minimum frequency count to filter out the patterns
        """

        pattern_freq_vec = {}
        mdocs = self.collection_patterns.find({'udocID': udocId}, {'_id':0, 'pattern':1, 'usentID': 1, 'weight':1}).sort('usentID', 1).batch_size(512)

        for mdoc in mdocs:
            
            pat = mdoc['pattern'].lower()
            freq_vec = self.collection_pattern_freq.find_one({'pattern': pat}) 
            
            # filter patterns' corpus frequency <= min_count 
            if not freq_vec:
                self.logging.warning('pattern freq of "%s" is not found' % (pat))
                continue
            elif sum(freq_vec['count'].values()) <= min_count:
                self.logging.warning('pattern freq of "%s" <= %d' % (pat, min_count))
                continue

            # build freq vector with all emotions
            weighted_freq_vec = {}
            for e in self.emotion_list:
                if e not in freq_vec['count']: 
                    freq_vec['count'][e] = 0.0
                weighted_freq_vec[e] = freq_vec['count'][e] * mdoc['weight']

            pattern_freq_vec[pat] = weighted_freq_vec
        import pdb; pdb.set_trace()
        return pattern_freq_vec



