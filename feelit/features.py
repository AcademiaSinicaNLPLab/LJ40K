# -*- coding: utf-8 -*-

##########################################
# classes:
#   feelit > features > FetchFile
#   feelit > features > FetchMongo
#
# function: 
#   fetch features from file
#
#   -MaxisKao @ 20140828
##########################################

import logging
from feelit import utils
import numpy as np


class FetchFile(object):
    """
    Fetch features from files
    usage:
        >> from feelit.features import FetchFile
        >> ff = FetchFile()
        >> ff.fetch()
    """
    def __init__(self, **kwargs):
        """
        Parameters:
            verbose: True/False
        """        
        loglevel = logging.DEBUG if 'verbose' in kwargs and kwargs['verbose'] == True else logging.INFO
        logging.basicConfig(format='[%(levelname)s] %(message)s', level=loglevel)        
         

class FetchMongo(object):
    """
    Fetch features from mongodb
    usage:
        >> from feelit.features import FetchMongo
        >> fm = FetchMongo()
        >> fm.fetch('TFIDF', '53a1921a3681df411cdf9f38')
        >> fm.tranform()
    """
    def __init__(self, **kwargs):
        """
        Parameters:
            verbose: True/False
        """
        loglevel = logging.DEBUG if 'verbose' in kwargs and kwargs['verbose'] == True else logging.INFO
        logging.basicConfig(format='[%(levelname)s] %(message)s', level=loglevel)

        self._db = None
        self._fetched = set()

        self.feature_dict_lst = []
        self.label_lst = []

        self.X = None
        self.y = None
        
    def _getCollectionName(self, feature_name, prefix="features"):
        return '.'.join([prefix, feature_name])

    def check(self, collection_name, setting_id):
        """
        1. check collection_name and setting_id
        2. automatically get valid setting_id if no setting_id specified

        Parameters:

        Returns: True/False
        """
        if (collection_name, setting_id) in self._fetched:
            return False

        ## check if collection_name if valid
        if collection_name not in self._db.collection_names():
            raise Exception('cannot find collection %s in %s' % (collection_name, self._db.name))
        else:
            ### the collection_name exists,
            ### check if setting_id is valid

            available_settings = self._db[collection_name].distinct('setting') 

            if setting_id in available_settings:
                logging.debug("use setting_id %s in collection %s" % (setting_id, collection_name) )
                return True
            else:
                logging.error("can't find setting_id %s in collection %s" % (setting_id, collection_name) )
                return False

            # if setting_id == "all" and len(available_settings) == 1:
            #     setting_id = available_settings[0]
            #     logging.info( "use setting %s automatically" % (setting_id) )

            # elif setting_id == "all" and len(available_settings) == 0:
            #     logging.info( "no filed named 'setting', will include all data")

            # if setting_id and setting_id in available_settings:
            #     logging.debug( "valid setting_id" )
            #     ## will return True
                
            # elif setting_id and setting_id not in available_settings:
            #     logging.warn( "setting %s doe'nt exists, specify a valid setting_id from %s" % (setting_id, ", ".join(available_settings)) )
            #     return False

            # elif setting_id == "all" and len(available_settings) > 1:
            #     logging.warn( "multiple settings found, specify a setting_id from %s" % ( ", ".join(available_settings) ) )
            #     return False

            # elif setting_id == "all" and len(available_settings) == 1:
            #     ## use first setting_id directly
            #     logging.info( "use setting %s automatically" % (setting_id) )
            #     setting_id = available_settings[0]
            #     ## will return True

            # elif not setting_id and len(available_settings) == 0:
            #     logging.info( "no filed named 'setting', will include all data")
            #     ## don't use setting_id in the query
            #     ## will return True

            # else:
            #     raise Exception("unknown error, check the input setting_id and collection_name")

            # self._setting_id = setting_id
            # return True

    def fetch(self, feature_name, setting_id, collection_name="auto", label_name="emotion"):

        """
        Load all added features from mongodb
        >> Parameters: 
            feature_name    : e.g., "TFIDF", "pattern_emotion", "keyword" ... etc.
            setting_id      : <str: mongo_ObjectId>, e.g., "53a1921a3681df411cdf9f38"
                              further version will support "all" / "first" / "random" / <str: mongo_ObjectId> 
            collection_name : "auto"/<str>, e.g. "features.TFIDF"
            label_name      : the field storing the target label in mongo, e.g., "emotion" or "emoID"

        >> Returns:
            (X, y): (<sparse matrix>, <array>)

        An example format of a document fetched from mongo:
        {
            "_id" : ObjectId("53a1922d3681df411cdf9f39"),
            "emotion" : "sleepy",
            "setting" : "53a1921a3681df411cdf9f38",
            "udocID" : 38000,

            ## list
            "feature" : 
            [
                [f1, ...], 
                [f2, ...],
            ]

            ## dict
            "feature" : 
            {
                f1: ...,
                f2: ...,
            }
        }
        """

        ## connect to mongodb if not connected
        if not self._db: 
            self._db = utils.connect_mongo()

        ## feature_name: TFIDF --> collection_name: TFIDF.features
        if collection_name in ("auto", None):
            collection_name = self._getCollectionName(feature_name)

        ## check if the current settings is valid
        if not self.check(collection_name, setting_id):
            return False

        if not setting_id:
            ## no setting_id, fetch all
            logging.debug( "no setting_id specified, fetch all from %s" % (collection_name) )
            cur = self._db[collection_name].find()
        else:
            ## with setting_id
            logging.debug( "fetch from %s with setting_id %s" % (collection_name, setting_id) )
            cur = self._db[collection_name].find({'setting': setting_id })

        _count = cur.count()
        logging.info("fetching %d documents from %s" % (_count, collection_name))
        
        for i, mdoc in enumerate(cur.batch_size(1024)):

            if 'feature' not in mdoc:
                logging.warn( "invalid format in the mongo document, skip this one." )
                continue

            ## get (and tranform) features into dictionary
            if type(mdoc['feature']) == dict:
                feature_dict = dict( mdoc['feature'] )

            elif type(mdoc['feature']) == list:
                feature_dict = { f[0]:f[1] for f in mdoc['feature'] }

            else:
                raise TypeError('make sure the feature format is either <dict> or <list> in mongodb')

            label = mdoc[label_name]

            self.feature_dict_lst.append( feature_dict )
            self.label_lst.append( label )

            logging.debug('mdoc %d/%d fetched' % ( i+1, _count))
            
        self._fetched.add( (collection_name, setting_id) )

    def tranform(self, reduce_memory=True):

        from sklearn.feature_extraction import DictVectorizer
        ## all feature_dict collected [ {...}, {...}, ..., {...} ]
        vec = DictVectorizer()
        self.X = vec.fit_transform( self.feature_dict_lst ) ## yield a sparse matrix
        self.y = np.array( self.label_lst )

        if reduce_memory:
            del self.feature_dict_lst
            del self.label_lst
            self.feature_dict_lst = []
            self.label_lst = []

        return (self.X, self.y)


# from sklearn.decomposition import TruncatedSVD as LSA
# lsa = LSA(n_components=100)
# _X = lsa.fit_transform(fm.X) ## _X: <40000x100>

class Fusion(object):
    """
    docstring for Fusion
    """
    def __init__(self, arg):
        pass
