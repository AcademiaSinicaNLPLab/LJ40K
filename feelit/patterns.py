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

    def get_patterns_by_udocId(self, udocId):

        """

        """
        mdocs = self.collection_patterns.find({'udocID': udocID}, {'_id':0, 'pattern':1, 'usentID': 1, 'weight':1}).batch_size(512)
        import pdb; pdb.set_trace()



