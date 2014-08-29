# -*- coding: utf-8 -*-

##########################################
# classes:
#   feelit > features > LoadFile
#   feelit > features > FetchMongo
#
# function: 
#   fetch features from file
#
#   -MaxisKao @ 20140828
##########################################

import logging, os
from feelit import utils
import numpy as np


class LoadFile(object):
    """
    Fetch features from files
    usage:
        >> from feelit.features import LoadFile
        >> lf = LoadFile(verbose=True)
        >> lf.load(path="...")
        >> lf.loads(root="/Users/Maxis/projects/emotion-detection-modules/dev/image/emotion_imgs_threshold_1x1_rbg_out_amend/out_f1")
        >> lf.concatenate()
        >> lf.dump(path="data/image_rgb_gist.Xy", ext="npz")
    """
    def __init__(self, **kwargs):
        """
        Parameters:
            verbose: True/False
        """        
        loglevel = logging.DEBUG if 'verbose' in kwargs and kwargs['verbose'] == True else logging.INFO
        logging.basicConfig(format='[%(levelname)s] %(message)s', level=loglevel)        
        
        self.Xs = {}
        self.ys = {}
        self.X = None
        self.y = None
        
    def load(self, path, label="auto", **kwargs):
        """
        input: <csv file> one feature per line
        output: <array> one document per line
        Parameters:
            path: path of a csv file
            label: "auto"/<str>, label of this data
            range
            kwargs:
                data_range: "all"/<int>: [0:]/[0:<int>]
                parameters of utils.load_csv()
        Returns:
            <np.array> [ [f1,...fn], [f1,...,fn],... ]
        """
        logging.debug('loading %s' % (path))
        data_range = "all" if not 'data_range' in kwargs else kwargs["data_range"]

        ## load csv files to <float> type
        lines = utils.load_csv(path, **kwargs)

        ## to numpy array and transpose
        X = np.array(lines).transpose()
        if type(data_range) == int and data_range < len(X):
            X = X[:data_range]

        ## assign label to this loaded data
        if label == "auto":
            label = path.split('/')[-1].split('.')[0].split('_')[0]

        y = np.array([label]*len(X))

        self.Xs[label] = X
        self.ys[label] = y

    def loads(self, root, ext=None, **kwargs):
        for fn in os.listdir(root):
            if ext and not fn.endswith(ext):
                continue
            else:
                self.load( path=os.path.join(root, fn), **kwargs)

    def concatenate(self):
        for label in self.Xs:
            if self.X == None: 
                self.X = np.array(self.Xs[label])
            else:
                self.X = np.concatenate((self.X, self.Xs[label]), axis=0)
            if self.y == None: 
                self.y = np.array(self.ys[label])
            else:
                self.y = np.concatenate((self.y, self.ys[label]), axis=0)

    def dump(self, path, ext="npz"):
        ## amend path
        path = path if not ext or path.endswith(ext) else path+ext 
        logging.debug("dumping X, y to %s" % (path))
        np.savez_compressed(path, X=self.X, y=self.y)

class FetchMongo(object):
    """
    Fetch features from mongodb
    usage:
        >> from feelit.features import FetchMongo
        >> fm = FetchMongo()
        >> fm.fetch('TFIDF', '53a1921a3681df411cdf9f38', data_range=800)
        >> fm.tranform()
        >> fm.dump(path="data/TFIDF.Xy", ext="npz")
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

    def fetch(self, feature_name, setting_id, collection_name="auto", label_name="emotion", data_range="all"):

        """
        Load all added features from mongodb
        >> Parameters: 
            feature_name    : e.g., "TFIDF", "pattern_emotion", "keyword" ... etc.
            setting_id      : <str: mongo_ObjectId>, e.g., "53a1921a3681df411cdf9f38"
                              further version will support "all" / "first" / "random" / <str: mongo_ObjectId> 
            collection_name : "auto"/<str>, e.g. "features.TFIDF"
            label_name      : the field storing the target label in mongo, e.g., "emotion" or "emoID"
            data_range      : "all"/<int>, used for fetching parts of data in each category

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

            cur = self._db[collection_name].find( { "$query":{}, "$orderby": { "udocID": 1 } } ).batch_size(1024)
            # cur = self._db[collection_name].find()
        else:
            ## with setting_id
            logging.debug( "fetch from %s with setting_id %s" % (collection_name, setting_id) )
            cur = self._db[collection_name].find( { "$query": {'setting': setting_id }, "$orderby": { "udocID": 1 } } ).batch_size(1024)

        _count = cur.count()
        logging.info("fetching %d documents from %s" % (_count, collection_name))
        
        for i, mdoc in enumerate(cur):

            if 'feature' not in mdoc:
                logging.warn( "invalid format in the mongo document, skip this one." )
                continue

            ## filter by data_range
            ## if data_range=800, then 0-799 will be kept
            if type(data_range) == int:
                if mdoc['ldocID'] >= data_range:
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
        """
        Dictionary --> Vectors <np.sparse_matrix>
        save variable in self.X, self.y
        """
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

    def dump(self, path, ext="npz"):
        ## amend path
        path = path if not ext or path.endswith(ext) else path+ext 
        logging.debug("dumping X, y to %s" % (path))
        np.savez_compressed(path, X=self.X, y=self.y)

# from sklearn.decomposition import TruncatedSVD as LSA
# lsa = LSA(n_components=100)
# _X = lsa.fit_transform(fm.X) ## _X: <40000x100>

class Fusion(object):
    """
    docstring for Fusion
    """
    def __init__(self, arg):
        pass
