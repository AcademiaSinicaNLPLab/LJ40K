# -*- coding: utf-8 -*-

##########################################
# classes:
#   feelit > features > PatternFetcher
#   feelit > features > FileSplitter
#   feelit > features > DataPreprocessor
#   feelit > features > LoadFile
#   feelit > features > FetchMongo
#   feelit > features > DimensionReduction
#   feelit > features > Fusion
#   feelit > features > Learning
#
#   -MaxisKao @ 20140828
##########################################

import logging, os, sys, pickle
from feelit import utils
import numpy as np

from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB, BaseNB
from sklearn.metrics import roc_curve, auc
from random import randint
import pymongo
from operator import add

'''
def load(path, fields="ALL"):
    """
    load `(X,y)`, `K` or `normal npz` format from a .npz file
    generally, this is wrapper for numpy.load() function

    Parameters
    ==========
        path: str
            path to the .npz file
        fields: list
            fields to load
    
    Returns
    =======
        1. (X, y)
        2. K
        3. numpy data
    """
    data = np.load(path)

    if "X" in data.files and ( type(fields) == list and "X" in fields or fields == "ALL" ):
        ## check X
        X = utils.toDense(data['X'])
        y = data['y']
        return (X,y)
    else:
        return data

def dump(path, **kwargs):
    """
    dump data into a numpy compressed file
    A wrapper for numpy.savez_compressed() function

    Parameters
    ==========
        path:
            Path to the .npz file
        kwargs:
            Arrays to save to the file
            e.g, dump('test.npz', X=X, y=y)
    """
    ## check data
    # try:        
    #     X, y = Xy
    # except ValueError:
    #     logging.error("check the format in Xy")
    #     return False

    ## check path
    dirs = os.path.dirname(path)
    if dirs and not os.path.exists( dirs ): os.makedirs( dirs )

    ## dump
    try:
        logging.debug("dumping X, y to %s" % (path))
    except NameError:
        pass
    np.savez_compressed(path, **kwargs)
'''

class PatternFetcher(object):
    """
    See batchFetchPatterns.py for example usage
    """

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

    def get_pattern_freq_by_udocId(self, udocId, min_count=1, weighted=True):

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

                w = mdoc['weight'] if weighted else 1.0
                weighted_freq_vec[e] = freq_vec['count'][e] * w

            pattern_freq_vec[pat] = weighted_freq_vec

        return pattern_freq_vec

    def _sum_pattern_vector(self, pf, use_score=False, vlambda=1.0):

        sum_vec = [0] * len(self.emotion_list)

        for freq_vec in pf.values():

            if use_score:
                score_vec = self.pattern_score(freq_vec, vlambda)
                temp_vec = score_vec.values()
            else:
                temp_vec = freq_vec.values()

            sum_vec = map(add, sum_vec, temp_vec)

        ## average the the pattern was proved to be worse
        #return [v/len(pf) for v in sum_vec] if len(pf) != 0 else sum_vec
        return sum_vec

    def sum_pattern_freq_vector(self, pf):
        """
        sum up pattern emotion arrays by occurence frequency
        """
        return self._sum_pattern_vector(pf, False)

    def sum_pattern_score_vector(self, pf, vlambda=1.0):
        """
        sum up pattern emotion score arrays
        """
        return self._sum_pattern_vector(pf, True, vlambda)

    def pattern_score(self, freq_vec, vlambda):
        """
        scoring a pattern emotion array
        """

        emotion_set = set(freq_vec.keys())
        S_vec = {}        
        for e in freq_vec:      # to keep the key order as same, we do not loop with set but dict

            ## s(p, e) = f(p, e)
            s_e = freq_vec[e]

            exclusive_set = emotion_set - set([e])
            sum_l1 = 0
            sum_l2 = 0

            for not_e in exclusive_set:             # here we don't care about the order
                sum_l1 += freq_vec[not_e]
                sum_l2 += pow(freq_vec[not_e], 2)

            ## beta =
            #               lambda ^ (L2_Norm(f(p, -e))
            #
            beta = pow(vlambda, pow(sum_l2, 0.5))

            ## s(p, -e) = 
            #                L2_Norm(f(p, -e)) ^ 2
            #              --------------------------
            #               L1_Norm(f(p, -e)) + beta
            #
            s_not_e = sum_l2 / (sum_l1+beta)

            ## final score S(p, e) = 
            #                       s(p, e)
            #               -----------------------
            #                 (s(p, e) + s(p, -e))
            #
            S_vec[e] = s_e / (s_e + s_not_e)

        return S_vec        


class FileSplitter(object):
    """
    see batchSplitEmotion.py for usage
    """

    def __init__(self, **kwargs):    

        if 'logger' in kwargs and kwargs['logger']:
            self.logging = kwargs['logger']
        else:
            logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.ERROR)  
            self.logging = logging

    def load(self, file_path):
        """
        parameters:
            file_path: input data path
        """
        data = np.load(file_path)

        self.X = data['X']
        self.y = data['y']

    def split(self, begin, end, samples_in_each_emotion=1000):
        """
        parameters:
            begin:
            end:
            samples_in_each_emotion:
        """

        if begin < 0 or end > samples_in_each_emotion:
            return False

        # we suppose that the input would be ordered by emotion with 1000 samples in each emotion
        n_emotion = self.X.shape[0]/samples_in_each_emotion

        self.X_sub = []
        self.y_sub = []

        for i in range(n_emotion):

            temp_begin = begin + i * samples_in_each_emotion
            temp_end = end + i * samples_in_each_emotion

            self.X_sub += self.X[temp_begin: temp_end].tolist()
            self.y_sub += self.y[temp_begin: temp_end].tolist()

    def _subsample_by_idx(self, X, y, idxs):
        """
        subsample a 2-D array by row index
        """
        _X, _y = [], []
        for i in idxs:
            _X.append(X[i])
            _y.append(y[i])

        return _X, _y

    def merge_negatives(self, idx_dict):
        """
        idx_dict: {'emotion': [(index, 'sample_emotion')], ...}
            i.e., {'tired': [(31201, 'tired'), (100, 'happy')]}
        """

        self.X_dict = {}
        self.y_dict = {}

        for i_label, label in enumerate(idx_dict):

            idxs = [i for i,l in idx_dict[label]]
            self.X_dict[label], self.y_dict[label] = self._subsample_by_idx(self.X_sub, self.y_sub, idxs)

    def _binary_label(self, y, emotion):
        return [1 if e == emotion else -1 for e in y]

    def dump_by_emotions(self, file_prefix, ext):
        """
        save self.X_dict to file_prefix_emotion.npz
        """
        for key, value in self.X_dict.iteritems():
            
            yb = self._binary_label(self.y_dict[key], key)

            # TODO: .train.npz is a hidden naming rule which should be eliminated
            fname = file_prefix + '.' + key + ext

            self.logging.debug("dumping X, y to %s" % (fname))
            np.savez_compressed(fname, X=np.array(value), y=np.array(yb))


    def dump(self, file_path, **kwargs):
        """
        parameters:
            file_path: output data path

        option:
            X: output data X
            y: output data y
        """
        out_X = self.X_sub if 'X' not in kwargs else kwargs['X']
        out_y = self.y_sub if 'y' not in kwargs else kwargs['y']

        self.logging.debug("dumping X, y to %s" % (file_path))
        np.savez_compressed(file_path, X=np.array(out_X), y=np.array(out_y))


class LoadFile(object):
    """
    Fetch features from files
    usage:
        >> from feelit.features import LoadFile
        >> lf = LoadFile(verbose=True)

        ## normal use: load all data
        >> lf.loads(root="/Users/Maxis/projects/emotion-detection-modules/dev/image/emotion_imgs_threshold_1x1_rbg_out_amend/out_f1")

        ## specify the data_range [:800]
        >> lf.loads(root="/Users/Maxis/projects/emotion-detection-modules/dev/image/emotion_imgs_threshold_1x1_rbg_out_amend/out_f1", data_range=(None,800) )

        ## amend value, ie., None -> 0, "NaN" -> 0
        >> lf.loads(root="/Users/Maxis/projects/emotion-detection-modules/dev/image/emotion_imgs_threshold_1x1_rbg_out_amend/out_f1", data_range=(None,800), amend=True)

        ## specify the data_range [-200:] and amend value
        >> lf.loads(root="/Users/Maxis/projects/emotion-detection-modules/dev/image/emotion_imgs_threshold_1x1_rbg_out_amend/out_f1", data_range=(-200,None), amend=True)

        >> lf.dump(path="data/image_rgb_gist.Xy", ext=".npz")
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

        ## replace None -> 0, -1 -> 0, "NaN" -> 0
        if "amend" in kwargs and kwargs["amend"] == True:
            logging.debug('amending %s' % (path))
            extra = {-1: 0, None: 0} if "extra" not in kwargs else kwargs["extra"]
            utils.all_to_float(lines, extra=extra)

        ## to numpy array and transpose
        X = np.array(lines).transpose()
        if type(data_range) == int and data_range < len(X):
            X = X[:data_range]

        elif type(data_range) == tuple or type(data_range) == list:

            _begin = data_range[0]
            _end = data_range[1]

            if _begin == None and _end == None:
                raise TypeError("data_range must contain at least one int value")

            elif _begin == None and _end != None:
                X = X[:_end]
            elif _begin != None and _end == None:
                X = X[_begin:]


        ## assign label to this loaded data
        if label == "auto":
            label = path.split('/')[-1].split('.')[0].split('_')[0]

        y = np.array([label]*len(X))

        self.Xs[label] = X
        self.ys[label] = y

    def loads(self, root, ext=None, **kwargs):
        fns = []
        for fn in os.listdir(root):
            if ext and not fn.endswith(ext):
                continue
            else:
                fns.append( fn )

        for fn in sorted(fns):
            self.load( path=os.path.join(root, fn), **kwargs)

        logging.debug("All loaded. Concatenate Xs and ys")
        self.concatenate()

    def concatenate(self):
        labels = sorted(self.Xs.keys())
        for label in labels:

            if self.X is None: 
                self.X = np.array(self.Xs[label])
            else:
                self.X = np.concatenate((self.X, self.Xs[label]), axis=0)

            if self.y is None: 
                self.y = np.array(self.ys[label])
            else:
                self.y = np.concatenate((self.y, self.ys[label]), axis=0)

    def dump(self, path, ext=".npz"):
        ## amend path
        path = path if not ext or path.endswith(ext) else path+ext 
        logging.debug("dumping X, y to %s" % (path))
        np.savez_compressed(path, X=self.X, y=self.y)

class FetchMongo(object):
    """
    Fetch features from mongodb
    usage:
        >> from feelit.features import FetchMongo
        >> fm = FetchMongo(verbose=True)

        ## set data_range to 800, i.e., [:800]
        >> fm.fetch_transform('TFIDF', '54129c359503bb27ce851ac4', data_range=800)

        ## set data_range > 800, i.e., [800:]
        >> fm.fetch_transform('TFIDF', '54129c359503bb27ce851ac4', data_range=">800")
        >> fm.dump(path="data/TFIDF.Xy", ext=".npz")
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
        
    def _get_collection_name(self, feature_name, prefix="features"):
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
            setting_id      : <str: mongo_ObjectId>, e.g., "54129c359503bb27ce851ac4"
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
            collection_name = self._get_collection_name(feature_name)

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

        logging.info("fetching documents from %s" % (collection_name))
        

        ## amend data_range
        ## ">800" --> 
        raw_data_range = data_range
        
        range_type = None
        if type(data_range) == int:
            range_type = "lt"
        elif raw_data_range == "all":
            range_type = "all"
        elif type(raw_data_range) == str and ">" in raw_data_range:
            range_type = "gte"
            data_range = int(data_range.replace(">", ""))
        else:
            logging.error("check the data_range format")
            return False


        for i, mdoc in enumerate(cur):

            if 'feature' not in mdoc:
                logging.warn( "invalid format in the mongo document, skip this one." )
                continue

            ## filter by data_range
            ## if data_range=800, then 0-799 will be kept
            if range_type == "lt":
                ## e.g., data_range=800
                if mdoc['ldocID'] >= data_range:
                    logging.debug('mdoc %d skipped' % ( i+1 ))
                    continue
            elif range_type == "gte":
                ## e.g., data_range=">800"
                if mdoc['ldocID'] < data_range:
                    logging.debug('mdoc %d skipped' % ( i+1 ))
                    continue
            elif range_type == "all":
                pass
            else:
                logging.error("unknown range_type")
                return False
            
            ## get (and tranform) features into dictionary
            if type(mdoc['feature']) == dict:
                feature_dict = dict( mdoc['feature'] )
            elif type(mdoc['feature']) == list:
                feature_dict = { f[0]:f[1] for f in mdoc['feature'] }
            else:
                raise TypeError('make sure the feature format is either <dict> or <list> in mongodb')

            ## reform the feature dictionary by setting up threshoukd
            feature_dict

            label = mdoc[label_name]

            self.feature_dict_lst.append( feature_dict )
            self.label_lst.append( label )

            logging.debug('mdoc %d fetched' % ( i+1 ))
            
        self._fetched.add( (collection_name, setting_id) )

    def tranform(self, reduce_memory=True, get_sparse=False):
        """
        Dictionary --> Vectors <np.sparse_matrix>
        save variable in self.X, self.y
        """
        from sklearn.feature_extraction import DictVectorizer
        ## all feature_dict collected [ {...}, {...}, ..., {...} ]
        vec = DictVectorizer(sparse=get_sparse)  # Douglas: try not to produce sparse matrix here. we will get things simpler
        self.X = vec.fit_transform( self.feature_dict_lst ) ## yield a sparse matrix. Douglas: ignore this comment
        self.y = np.array( self.label_lst )

        if reduce_memory:
            del self.feature_dict_lst
            del self.label_lst
            self.feature_dict_lst = []
            self.label_lst = []

        return (self.X, self.y)

    def fetch_transform(self, feature_name, setting_id, collection_name="auto", label_name="emotion", data_range="all", reduce_memory=True):
        """
        a wrapper of fetch() and transform()
        """
        self.fetch(feature_name, setting_id, collection_name=collection_name, label_name=label_name, data_range=data_range)
        self.tranform(reduce_memory=reduce_memory)

    def dump(self, path, ext=".npz"):
        ## amend path
        path = path if not ext or path.endswith(ext) else path+ext 
        logging.debug("dumping X, y to %s" % (path))
        np.savez_compressed(path, X=self.X, y=self.y)

class DimensionReduction(object):   
    """
    DimensionReduction: wrapper of LSA, ICA in scikit-learn

    usage:
        >> from feelit.features import DimensionReduction
        >> dr = DimensionReduction(algorithm="LSA")
        >> _X = dr.reduction(X, n_components=40)

    Parameters
    ==========
        algorithm : str
            support "TruncatedSVD" (or "LSA"), FastICA (or "ICA")
    """
    def __init__(self, algorithm):
        algo = algorithm.strip().lower()
        if algo in ("truncatedsvd", "lsa"):
            self.algorithm = "LSA"
        elif algo in ("fastica", "ica"):
            self.algorithm = "ICA"

    def reduction(self, X, n_components):
        if self.algorithm == "LSA":
            from sklearn.decomposition import TruncatedSVD as LSA
            worker = LSA(n_components=n_components)
        elif self.algorithm == "ICA":
            from sklearn.decomposition import FastICA as ICA
            worker = ICA(n_components=n_components)

        _X = worker.fit_transform(X)
        return _X

class DataPreprocessor(object):
    """
    Fuse features from .npz files
    usage:
        >> from feelit.features import DataPreprocessor
        >> import json
        >> features = ['TFIDF', 'keyword', 'xxx', ...]
        >> dp = DataPreprocessor()
        >> dp.loads(features, files)
        >> X, y = dp.fuse()
    """
    def __init__(self, **kwargs):
        """
        options:
            logger: logging instance
        """
        self.clear()

        if 'logger' in kwargs and kwargs['logger']:
            self.logging = kwargs['logger']
        else:
            logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.ERROR)  
            self.logging = logging
    '''
    def full_matrix(self, X):
        """
        Return full_matrix(X)

        if self.X is in sparse representation, 
            transform it back to full representation
        if not,
            do nothing

        we check isSparse by checking if it is possible to get the list len()
        """
        if X is not None and utils.isSparse(X):
            self.logging.info('sparse matrix representation detected, transform it to full representation')
        return utils.toDense(X)

    def replace_nan(self, X, NaN=0.0, NONE=0.0):
        """
        replace string, NaN, None to 0.0
        """
        if X is not None:
            for i in xrange(len(X)):
                for j in xrange(len(X[i])):
                    if type(X[i][j]) == float:
                        continue
                    else:                        
                        if X[i][j] is None:
                            self.logging.debug('element None detected in X[%d][%d]' % (i,j))
                            X[i][j] = NONE
                        elif X[i][j] == 'NaN':
                            self.logging.debug('element "NaN" detected in X[%d][%d]' % (i,j))
                            X[i][j] = NaN
                        elif type(X[i][j]) == int:
                            X[i][j] = float(X[i][j]) 
                        else:
                            X[i][j] = NONE
        return X
    '''
    def loads(self, features, paths):
        """
        Input:
            paths       : list of files to be concatenated
            features:   : list of feature names
        """
        for i, path in enumerate(paths):
            self.logging.info('loading data from %s' % (path))
            data = np.load(path)            

            #X =  self.replace_nan( self.full_matrix(data['X']) )
            X = data['X']
            self.Xs[features[i]] = X
            self.ys[features[i]] = data['y'];

            self.logging.info('feature "%s", %dx%d' % (features[i], X.shape[0], X.shape[1]))

            self.feature_name.append(features[i])

    def fuse(self):
        """
        Output:
            fused (X, y) from (self.Xs, self.ys)
        """

        # try two libraries for fusion
        try:
            X = np.concatenate(self.Xs.values(), axis=1)
        except ValueError:
            from scipy.sparse import hstack
            candidate = tuple([arr.all() for arr in self.Xs.values()])
            X = hstack(candidate)
              
        y = self.ys[ self.ys.keys()[0] ]
        # check all ys are same  
        for k, v in self.ys.items():
            assert (y == v).all()
        feature_name = '+'.join(self.feature_name)

        self.logging.debug('fused feature name is "%s", %dx%d' % (feature_name, X.shape[0], X.shape[1]))

        return X, y, feature_name

    def clear(self):
        self.Xs = {}
        self.ys = {}
        self.feature_name = []

    def get_binary_y_by_emotion(self, y, emotion):
        '''
        return y with elements in {1,-1}
        '''       
        yb = np.array([1 if val == emotion else -1 for val in y])
        return yb

class Learning(object):
    """
    usage:
        >> from feelit.features import Learning
        >> learner = Learning(verbose=args.verbose, debug=args.debug) 
        >> learner.set(X_train, y_train, feature_name)
        >>
        >> scores = {}
        >> for C in Cs:
        >>  for gamma in gammas:
        >>      score = learner.kFold(kfolder, classifier='SVM', 
        >>                          kernel='rbf', prob=False, 
        >>                          C=c, scaling=True, gamma=gamma)
        >>      scores.update({(c, gamma): score})
        >>
        >> best_C, best_gamma = max(scores.iteritems(), key=operator.itemgetter(1))[0]
        >> learner.train(classifier='SVM', kernel='rbf', prob=True, C=best_C, gamma=best_gamma, 
        >>              scaling=True, random_state=np.random.RandomState(0))
        >> results = learner.predict(X_test, yb_test, weighted_score=True, X_predict_prob=True, auc=True)
    """

    def __init__(self, X=None, y=None, **kwargs):

        if 'logger' in kwargs and kwargs['logger']:
            self.logging = kwargs['logger']
        else:
            logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.ERROR)  
            self.logging = logging    

        self.X = X
        self.y = y
        self.kfold_results = []
        self.Xs = {}
        self.ys = {}

    def set(self, X, y, feature_name):
        self.X = X
        self.y = y
        self.feature_name = feature_name

    def train(self, **kwargs):
        self._train(self.X, self.y, **kwargs)

    def _train(self, X_train, y_train, **kwargs):
        """
        required:
            X_train, y_train

        options:
            classifier: 'SVM', 'SGD', 'GaussianNB'
            with_mean: True/False
            with_std: True/False
            scaling: True/False
            prob: True/False. Esimate probability during training
            random_state: seed, RandomState instance or None; for probability estimation
            kernel: 'rbf', ...
            C: float; svm parameters
            shuffle: True/False; for SGD
        """
        ## setup a classifier
        classifier = "SVM" if "classifier" not in kwargs else kwargs["classifier"]

        # ## slice 
        # delete = None if "delete" not in kwargs else kwargs["delete"]

        # if delete:
        #     X_train = np.delete(utils.toDense(self.X), delete, axis=0)
        #     y_train = np.delete(self.y, delete, axis=0)
        # else:

        self.logging.debug("%d samples x %d features in X_train" % ( X_train.shape[0], X_train.shape[1] ))
        self.logging.debug("%d samples in y_train" % ( y_train.shape[0] ))

        with_mean = True if 'with_mean' not in kwargs else kwargs['with_mean']
        with_std = True if 'with_std' not in kwargs else kwargs['with_std']

        # Cannot center sparse matrices, `with_mean` should be set as `False`
        # Douglas: this doesn't make sense
        #if utils.isSparse(self.X):
        #    with_mean = False
        
        self.scaling = False if 'scaling' not in kwargs else kwargs['scaling']
        if self.scaling:
            self.scaler = StandardScaler(with_mean=with_mean, with_std=with_std)
            ## apply scaling on X
            self.logging.debug("applying a standard scaling with_mean=%d, with_std=%d" % (with_mean, with_std))
            X_train = self.scaler.fit_transform(X_train)

        ## determine whether using predict or predict_proba
        self.prob = False if 'prob' not in kwargs else kwargs["prob"]
        random_state = None if 'random_state' not in kwargs else kwargs["random_state"]
        
        if classifier == "SVM":
            ## setup a svm classifier
            kernel = "rbf" if 'kernel' not in kwargs else kwargs["kernel"]
            ## cost: default 1
            C = 1.0 if "C" not in kwargs else kwargs["C"]
            ## gamma: default (1/num_features)
            num_features = X_train.shape[1]
            gamma = (1.0/num_features) if "gamma" not in kwargs else kwargs["gamma"]
            #self.clf = svm.SVC(C=C, gamma=gamma, kernel=kernel, probability=self.prob, random_state=random_state, class_weight='auto')
            self.clf = svm.SVC(C=C, gamma=gamma, kernel=kernel, probability=self.prob, random_state=random_state)
            self.params = "%s_%s C=%f gamma=%f probability=%d" % (classifier, kernel, C, gamma, self.prob)

        elif classifier == "SGD":

            shuffle = True if 'shuffle' not in kwargs else kwargs['shuffle']
            if self.prob:
                self.clf = SGDClassifier(loss="log", shuffle=shuffle)
            else:
                self.clf = SGDClassifier(shuffle=shuffle)

            self.params = "%s_%s" % (classifier, 'linear')
        elif classifier == "GaussianNB":
            self.clf = GaussianNB()

            self.params = "%s_%s" % (classifier, 'NB')
        else:
            raise Exception("currently only support SVM, SGD and GaussianNB classifiers")

        self.logging.debug(self.params)
        self.clf.fit(X_train, y_train)
    
    def predict(self, X_test, y_test, **kwargs):
        '''
        return dictionary of results
        '''

        if self.scaling:
            X_test = self.scaler.transform(X_test)

        self.logging.info('y_test = %s', str(y_test.shape))
        y_predict = self.clf.predict(X_test)
        X_predict_prob = self.clf.predict_proba(X_test) if self.prob else 0
        results = {}
        if 'score' in kwargs and kwargs['score'] == True:
            results.update({'score': self.clf.score(X_test, y_test.tolist())})
            self.logging.info('score = %f', results['score'])

        if 'weighted_score' in kwargs and kwargs['weighted_score'] == True:
            results.update({'weighted_score': self._weighted_score(y_test.tolist(), y_predict)})
            self.logging.info('weighted_score = %f', results['weighted_score'])

        if 'y_predict' in kwargs and kwargs['y_predict'] == True:
            results.update({'y_predict': y_predict})
            self.logging.info('y_predict = %f', results['y_predict'])

        if 'X_predict_prob' in kwargs and kwargs['X_predict_prob'] == True:            
            results.update({'X_predict_prob': X_predict_prob[:, 1]})
            self.logging.info('X_predict_prob = %s', str(results['X_predict_prob']))

        if 'auc' in kwargs and kwargs['auc'] == True:
            fpr, tpr, thresholds = roc_curve(y_test, X_predict_prob[:, 1])
            results.update({'auc': auc(fpr, tpr)})
            self.logging.info('auc = %f', results['auc'])

        return results     
    
    def _weighted_score(self, y_test, y_predict):
        # calc weighted score 
        n_pos = len([val for val in y_test if val == 1])
        n_neg = len([val for val in y_test if val == -1])
        
        temp_min = min(n_pos, n_neg)
        weight_pos = 1.0/(n_pos/temp_min)
        weight_neg = 1.0/(n_neg/temp_min)
        
        correct_predict = [i for i, j in zip(y_test, y_predict) if i == j]
        weighted_sum = 0.0
        for answer in correct_predict:
            weighted_sum += weight_pos if answer == 1 else weight_neg
        
        wscore = weighted_sum / (n_pos * weight_pos + n_neg * weight_neg)
        return wscore
    
    def kfold(self, kfolder, **kwargs):
        """
        return:
            mean score for kfold training

        required:
            kfolder: generated by sklearn.cross_validatio.KFold

        options:
            same as _train
        """
        
        #amend = False if "amend" not in kwargs else kwargs["amend"]
        #if amend:
            ## amend dense matrix: replace NaN and None with float values
        #    self.check_and_amend()
        #else:
        #    self.logging.debug("skip the amending process")

        sum_score = 0.0
        for (i, (train_index, test_index)) in enumerate(kfolder):

            self.logging.info("cross-validation fold %d: train=%d, test=%d" % (i, len(train_index), len(test_index)))

            X_train, X_test, y_train, y_test = self.X[train_index], self.X[test_index], self.y[train_index], self.y[test_index]
            self._train(X_train, y_train, **kwargs)

            score = self.predict(X_test, y_test, score=True)['score']
            self.logging.info('score = %.5f' % (score))
            sum_score += score

        mean_score = sum_score/len(kfolder)
        self.logging.info('*** C = %f, mean_score = %f' % (kwargs['C'], mean_score))
        return mean_score



    '''  
    def slice(self, each_class):

        if "<" in each_class:
            th = int(each_class.replace("<",""))
            to_delete = [gidx for gidx, lidx in self.idx_map.iteritems() if lidx >= th]
        elif ">" in each_class: # e.g., >800, including 800
            th = int(each_class.replace(">",""))
            to_delete = [gidx for gidx, lidx in self.idx_map.iteritems() if lidx < th]
        else:
            self.logging.error('usage: e.g., l.slice(each_class=">800")')
            return False

        return to_delete        
        # X_ = np.delete(self.X, to_delete, axis=0)
        # y_ = np.delete(self.y, to_delete, axis=0)

    def load(self, path):
        # fn: DepPairs_LSA512+TFIDF_LSA512+keyword_LSA512+rgba_gist+rgba_phog.npz
        # fn: image_rgb_gist.Xy.npz
        data = np.load(path)
        self.X = data['X']
        self.y = data['y']

        self.feature_name = path.split('/')[-1].replace('.Xy','').replace(".train","").replace(".test","").split('.npz')[0]

        ## build global idx --> local idx mapping
        # lid, prev = 0, self.y[0]
        # self.idx_map = {}
        # for i,current in enumerate(self.y):
        #     if prev and prev != current:
        #         prev = current
        #         lid = 0
        #     self.idx_map[i] = lid
        #     lid += 1

    def save(self, root="results"):
        
        # self.clf.classes_ ## corresponding label of the column in predict_results
        # self.predict_results ## predict probability over 40 classes
        # self.y ## answers in testing data

        out_path = os.path.join(root, self.feature_name+".res.npz" )
        np.savez_compressed(out_path, tests=self.y, predicts=self.predict_results, classes=self.clf.classes_ )

    def save_kFold(self, root=".", feature_name="", ext=".npz"):
        if not self.feature_name:
            if feature_name:
                self.feature_name = feature_name
            else:
                self.logging.warn("speficy the feature_name for the file to be saved")
                return False

        tests, predicts, scores, classes = [], [], [], []
        for i, y_test, result, score, cla in self.kfold_results:
            tests.append( y_test )
            predicts.append( result )
            scores.append( score )
            classes.append( cla )


        out_path = os.path.join(root, self.feature_name+".res"+ext )

        if not os.path.exists(os.path.dirname(out_path)): os.makedirs(os.path.dirname(out_path))

        np.savez_compressed(out_path, tests=tests, predicts=predicts, scores=scores, classes=classes)

    def save_files(self, root=".", feature_name=""):
        """
        """
        if not self.feature_name:
            if feature_name:
                self.feature_name = feature_name
            else:
                self.logging.warn("speficy the feature_name for the file to be saved")
                return False

        subfolder = self.feature_name

        for ith, y_test, result, score in self.kfold_results:

            out_fn = "%s.fold-%d.result" % (self.feature_name, ith)

            ## deal with IOError
            ## IOError: [Errno 2] No such file or directory: '../results/text_TFIDF_binary/text_TFIDF.accomplished/text_TFIDF.accomplished.fold-1.result'
            out_path = os.path.join(root, subfolder, out_fn)
            out_dir = os.path.dirname(out_path)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            with open(out_path, 'w') as fw:

                fw.write( ','.join( map(lambda x:str(x), y_test) ) )
                fw.write('\n')

                fw.write( ','.join( map(lambda x:str(x), result) ) )
                fw.write('\n')


    def save_model(self, root=".", feature_name="", ext=".model"):
        
        if not self.feature_name:
            if feature_name:
                self.feature_name = feature_name
            else:
                self.logging.warn("speficy the feature_name for the file to be saved")
                return False
        out_path = os.path.join(root, self.feature_name+"."+self.params+".model" )
        out_dir = os.path.dirname(out_path)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        pickle.dump(self.clf, open(out_path, "wb"), protocol=2)
        self.logging.info("dump model to %s" %(out_path))

        return out_path

    def load_model(self, path):
        self.logging.info("loading model from %s" % (path))
        self.clf = pickle.load(open(path))



class Fusion(object):
    """
    Fusion features from .npz files
    usage:
        >> from feelit.features import Fusion
        >> fu = Fusion(verbose=True)
        >> fu.loads(a1, a2, ...)
        >> fu.fuse()
        >> fu.dump()
    """
    def __init__(self, *args, **kwargs):
        """
        Fusion( a1, a1, ..., verbose=True)
        Parameters
        ----------
        a1, a2, ... : str
            files to be fused
        verbose : boolean, optional
            True/False, set True to log debug level message
        """
        loglevel = logging.DEBUG if 'verbose' in kwargs and kwargs['verbose'] == True else logging.INFO
        logging.basicConfig(format='[%(levelname)s] %(message)s', level=loglevel)

        self.Xs = {}
        self.ys = {}

        self.X = None
        self.y = None

    def load(self, path):
        """
        load(path)

        load single file to fuse

        Parameters
        ----------
        path : str
            path to the candidate file

        Returns
        -------
        status : boolean
            True : added
            False : not added
        """

        ## extract filename to be the feature_name (key) in Xs and ys
        fn = path.split('/')[-1].split('.npz')[0]

        if fn in self.Xs or fn in self.ys:
            logging.info('feature %s already exists' % (fn))
            return False
        else:

            
            data = np.load(path)

            logging.info('loading %s into self.Xs[%s], self.ys[%s]' % (path, fn, fn))
            ## get shape info
            X_shape = utils.getShape(data['X'])
            y_shape = utils.getShape(data['y'])

            # logging.debug('loading X in %s...' % (fn))
            self.Xs[fn] = data['X']

            logging.debug("%s.X: <%d x %d> Loaded" % (fn, X_shape[0], X_shape[1]))

            # logging.debug('loading y in %s...' % (fn))
            self.ys[fn] = data['y']
            logging.debug("%s.y: <%d x 1> Loaded" % (fn, y_shape[0]))

            return True

    def loads(self, *args):
        """
        loads(a1, a2, ...)

        load multiple files to fuse

        Parameters
        ----------
        a1, a2, ... : str
            files to be fused
        """
        for fn in args:
            self.load(fn)

    def fuse(self, reduce_memory=False, override=False):
        """
        fuse loaded feature arrays

        Parameters
        ----------
        reduce_memory : boolean
            set True to del self.Xs once complete fusion
        override : boolean
            set True to override previous fusion results

        Returns
        -------
        status : boolean
            True: fuse successfully
            False: no fusion performed
        """
        logging.info('fusing %s ...' % (', '.join(self.Xs.keys())))

        if (not self.X or not self.y) and override:
            logging.warn('set override=True to override current fusion results')
            return False
        else:

            logging.debug('fusing X')

            ## detect dense or sparse
            try:
                ## for dense
                logging.debug("trying to concatenate matrix using numpy.concatenate")
                self.X = np.concatenate(self.Xs.values(), axis=1)
            
            except ValueError:
                ## to dense
                logging.debug("sparse matrix detected. Use scipy.sparse.hstack() instead to enhance performance")
                from scipy.sparse import hstack
                candidate = tuple([arr.all() for arr in self.Xs.values()])
                self.X = hstack(candidate)
                
            if reduce_memory: del self.Xs

            logging.debug('fusing y')
            self.y = self.ys[ self.ys.keys()[0] ]

            ## print infomation
            logging.info("Yield a fused matrix in shape (%d, %d)" % (self.X.shape[0], self.X.shape[1]))
            return True

    def dump(self, root="data", path="auto", ext=".npz"):

        ## text_DepPairs_LSA512.Xy
        ## '_'.join(x.split('.Xy')[0].split('_')[1:]) --> DepPairs_LSA512
        if path == "auto":
            path = '+'.join(sorted([ '_'.join(x.split('.Xy')[0].split('_')[1:]) for x in self.Xs.keys() ])) + ".Xy"

        ## amend path
        path = path if not ext or path.endswith(ext) else path+ext

        path = os.path.join(root, path)

        dirs = os.path.dirname(path)
        if dirs and not os.path.exists( dirs ): os.makedirs( dirs )

        logging.debug("dumping X, y to %s" % (path))
        np.savez_compressed(path, X=self.X, y=self.y)


    '''
