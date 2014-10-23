# -*- coding: utf-8 -*-

##########################################
# classes:
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
import pickle

__LIBSVM__ = "/tools/libsvm/python"

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

            if self.X == None: 
                self.X = np.array(self.Xs[label])
            else:
                self.X = np.concatenate((self.X, self.Xs[label]), axis=0)

            if self.y == None: 
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
    # from sklearn.decomposition import TruncatedSVD as LSA
    # lsa = LSA(n_components=512)
    # _X = lsa.fit_transform(fm.X) ## _X: <40000x100>
    # np.savez_compressed('data/TFIDF_LSA512.Xy.npz', X=_X, y=y)

    # from sklearn.decomposition import FastICA as ICA
    # ica = ICA(n_components=40)
    # _X = ica.fit_transform(fm.X)
    # np.savez_compressed('data/TFIDF_ICA40.Xy.npz', X=_X, y=y)    
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

class LIBSVM(object):
    """
    currently only supports binary version
    i.e., the label should be either `happy` or `_happy`

    Usage:

    from feelit.features import LIBSVM
    svm = LIBSVM(verbose=True)

    ####################################
    ## load training samples
    svm.load_train("data/text_TFIDF.Xy.train.npz")

    ## set param for training
    svm.set_param("-t 0 -c 4 -b 1 -q")

    ## formulate problem
    svm.formulate()

    ## training
    svm.train()

    ## save model
    svm.save_model(root="models", filename="auto")

    ####################################

    ## load_model

    ## load testing samples
    svm.load_test("data/image_rgb_phog.Xy.test.npz")

    ## set param for testing
    svm.set_param("-b 1 -q")

    ## testing
    svm.predict()


    """
    if not os.path.exists(__LIBSVM__):
        logging.error("can't find the LIBSVM in %s" % (__LIBSVM__))
    
    sys.path.append(__LIBSVM__)

    import svmutil

    def __init__(self, **kwargs):
        self.verbose = True if 'verbose' in kwargs and kwargs['verbose'] == True else False
        loglevel = logging.DEBUG if self.verbose else logging.INFO
        logging.basicConfig(format='[%(levelname)s] %(message)s', level=loglevel)

        self.y_train, self.X_train, self.y_test, self.X_test = None, None, None, None
        self.feature_name = None
        self.label_map = None

    # def build_numeric_label_map(self, y):
    #     logging.debug("building label numeric mapping")
    #     self.label_map = {_y:i for i,_y in enumerate(sorted(list(set(y))))}
    #     pickle.dump(self.label_map, open("numeric_label_map.pkl", "wb"))
    # def numerize(self, y, label_map=None):
    #     if not label_map and not self.label_map:
    #         self.build_numeric_label_map(y)
    #     if not label_map:
    #         label_map = self.label_map
    #     return [label_map[_y] for _y in y]

    def numerize(self, y):
        ############################## future work ##############################
        if len(set(y)) != 2:
            raise Exception("currently only supports binary classifier")
        #########################################################################

        return [0 if _y.startswith("_") else 1 for _y in y]
        
    def scale(self, X, with_mean=True, with_std=True):
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler(with_mean=with_mean, with_std=with_std)
        return scaler.fit_transform(X)

    def load_train(self, path, scaling=False):
        data = np.load(path)

        _scaling_msg_ = "and scaling " if scaling else ""
        logging.debug("transforming %s X to list" % (_scaling_msg_))
        self.X_train = data['X'].tolist() if not scaling else self.scale(data['X']).tolist()
        
        logging.debug("transforming y to list")
        self.y_train = data['y'].tolist()

        ## get feature_name
        self.feature_name = path.split('/')[-1].replace('.Xy','').split('.npz')[0]
        logging.debug("got feature_name %s" % (self.feature_name))

        ## if y_train looks like "happy" or "_happy"
        ## transform it to numeric expression such as 1 or 0
        if type(self.y_train[0]) not in (int , float):
            self.y_train = self.numerize(y=self.y_train)

    def load_test(self, path, scaling=False):

        data = np.load(path)

        _scaling_msg_ = "and scaling " if scaling else ""
        logging.debug("transforming %s X to list" % (_scaling_msg_))
        self.X_test = data['X'].tolist() if not scaling else self.scale(data['X']).tolist()
        
        logging.debug("transforming y to list")
        self.y_test = data['y'].tolist()
        

        self.feature_name = path.split('/')[-1].replace('.Xy','').split('.npz')[0]
        
        logging.debug("got feature_name %s" % (self.feature_name))

    def set_param(self, args="-t 0 -c 4 -b 1 -q"):
        self.param = self.svmutil.svm_parameter(args)
        return self.param

    def formulate(self):
        if not self.y_train or not self.X_train:
            logging.error("run load_train() to load training instances")
            return False
        self.prob = self.svmutil.svm_problem(self.y_train, self.X_train)

    def train(self, **kwargs):
        self.m = self.svmutil.svm_train(self.prob, self.param)

    def _build_classes_order(self, p_vals, p_labels, target):
        # e.g.,
        #  if p_val [0.53, 0.47] is labeled as p_label 0, then the order is [ 0, 1 ] --> [_happy, happy]
        #  if p_val [0.53, 0.47] is labeled as p_label 1, then the order is [ 1, 0 ] --> [happy, _happy]        
        #  if p_val [0.43, 0.57] is labeled as p_label 1, then the order is [ 0, 1 ] --> [_happy, happy]
        order = None
        for i in xrange(len(p_labels)):
            
            p_val, p_label = p_vals[i], p_labels[i]
            
            if p_val[0] == p_val[1]:
                continue
            else:
                maxidx = 0 if p_val[0] > p_val[1] else 1
                if maxidx == p_label:
                    order = [0, 1]
                    break
                else:
                    order = [1, 0]
                    break
        if not order :
            return False
        else:
            ## if the order is [ 0, 1 ]
            ## and the target is 'happy'
            ## then will return [ '_happy', 'happy' ]
            return [ "_"+target if _class == 0 else target for _class in order ]

    def predict(self, target, param=""):
        if not self.y_test or not self.X_test:
            logging.error("run load_test() to load training instances")
            return False

        ## relabel y_test
        logging.debug("relabeling y_test")
        ny_test = [1 if _y == target else 0 for _y in self.y_test]

        ## p_label, p_acc, p_val = svm_predict(y, x, m, '-b 1')
        logging.debug("predict X_test with options: %s" % (param))
        self.p_labels, self.p_acc, self.p_vals = self.svmutil.svm_predict(y=ny_test, x=self.X_test, m=self.m, options=param)
        
        # build classes order
        ## e.g., 
        #  if p_val [0.53, 0.47] is labeled as p_label 0, then the order is [ 0, 1 ] --> [_happy, happy]
        #  if p_val [0.53, 0.47] is labeled as p_label 1, then the order is [ 1, 0 ] --> [happy, _happy]
        self.classes_ = self._build_classes_order(self.p_vals, self.p_labels, target)

    def save_model(self, filename="auto", root="models", m=None):
        ## use abs "models" path instead
        path = os.path.join(root, ".".join([self.feature_name, "model"])) if filename == "auto" else os.path.join(root, filename)
        dirs = os.path.dirname(path)
        if dirs and not os.path.exists( dirs ): os.makedirs( dirs )  

        if not m: m = self.m
        self.svmutil.svm_save_model(path, m)

    def load_model(self, path):
        self.m = self.svmutil.svm_load_model(path)

class Learning(object):
    """
    usage:
        >> from feelit.features import Learning
        >> l = Learning(verbose=True)
        >> l.load(path="data/image_rgba_gist.Xy.npz")

        ## normal training/testing
        >> to_delete = l.slice(each_class=">800")
        >> l.train(classifier="SVM", delete=to_delete, kernel="rbf", prob=True)
        >> l.save_model()
        >> l.test()
        
        ## n-fold
        >> l.kFold(classifier="SVM")

        ## save n-fold result
        >> l.save(root="results")
    """


    def __init__(self, X=None, y=None, **kwargs):

        loglevel = logging.DEBUG if 'verbose' in kwargs and kwargs['verbose'] == True else logging.INFO
        logging.basicConfig(format='[%(levelname)s] %(message)s', level=loglevel)     

        self.X = X
        self.y = y
        self.kfold_results = []

    def load(self, path):
        # fn: DepPairs_LSA512+TFIDF_LSA512+keyword_LSA512+rgba_gist+rgba_phog.npz
        # fn: image_rgb_gist.Xy.npz
        data = np.load(path)
        self.X = data['X']
        self.y = data['y']



        self.feature_name = path.split('/')[-1].replace('.Xy','').replace(".train","").replace(".test","").split('.npz')[0]
        self.fn = self.feature_name

        ## build global idx --> local idx mapping
        # lid, prev = 0, self.y[0]
        # self.idx_map = {}
        # for i,current in enumerate(self.y):
        #     if prev and prev != current:
        #         prev = current
        #         lid = 0
        #     self.idx_map[i] = lid
        #     lid += 1


    def nFold(self, **kwargs):
        self.KFold(**kwargs)

    def check_and_amend(self, NaN=0.0, NONE=0.0):
        """
        - deal with transformation of sparse matrix format
            i.e., Reassign X by X = X.all(): because of numpy.load()
        - replace NaN and None if a dense matrix is given
        """
        logging.debug("start check and amend matrix X")

        if self.X != None and self.y != None:

            if utils.isSparse(self.X):
                ## if X is a sparse array --> it came from DictVectorize
                ## Reassign X by X = X.all(): because of numpy.load()
                self.X = self.X.all()
                logging.debug("sparse matrix detected.")
                logging.debug("no need to further amending on a sparse matrix")
                return False
            else:
                logging.debug('dense matrix detected')
                logging.debug("start to check and amend matrix value, fill NaN with %f and None with %f" % (NaN, NONE))
                replaced = 0
                for i in xrange(len(self.X)):
                    for j in xrange(len(self.X[i])):
                        if type(self.X[i][j]) == float:
                            continue
                        else:
                            replaced += 1
                            if self.X[i][j] == None:
                                self.X[i][j] = NONE
                            elif self.X[i][j] == 'NaN':
                                self.X[i][j] = NaN
                            else:
                                self.X[i][j] = NONE
                self._checked = True
                return replaced
        else:
            logging.debug("no X to be checked and amended")
            return False
  
    def slice(self, each_class):

        if "<" in each_class:
            th = int(each_class.replace("<",""))
            to_delete = [gidx for gidx, lidx in self.idx_map.iteritems() if lidx >= th]
        elif ">" in each_class: # e.g., >800, including 800
            th = int(each_class.replace(">",""))
            to_delete = [gidx for gidx, lidx in self.idx_map.iteritems() if lidx < th]
        else:
            logging.error('''usage: e.g., l.slice(each_class=">800")''')
            return False

        return to_delete        
        # X_ = np.delete(self.X, to_delete, axis=0)
        # y_ = np.delete(self.y, to_delete, axis=0)

    def train(self, **kwargs):
        from sklearn import svm
        from sklearn.linear_model import SGDClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB, BaseNB

        ## setup a classifier
        classifier = "SGD" if "classifier" not in kwargs else kwargs["classifier"]

        # ## slice 
        # delete = None if "delete" not in kwargs else kwargs["delete"]

        # if delete:
        #     X_train = np.delete(utils.toDense(self.X), delete, axis=0)
        #     y_train = np.delete(self.y, delete, axis=0)
        # else:
        X_train = self.X
        y_train = self.y

        logging.debug("%d samples x %d features in X_train" % ( X_train.shape[0], X_train.shape[1] ))
        logging.debug("%d samples in y_train" % ( y_train.shape[0] ))

        with_mean = True if 'with_mean' not in kwargs else kwargs['with_mean']
        with_std = True if 'with_std' not in kwargs else kwargs['with_std']

        # Cannot center sparse matrices, `with_mean` should be set as `False`
        if utils.isSparse(self.X):
            with_mean = False

        scaling = False if 'scaling' not in kwargs else kwargs['scaling']
        if scaling:
            scaler = StandardScaler(with_mean=with_mean, with_std=with_std)
            ## apply scaling on self.X
            logging.debug("applying a standard scaling")
            X_train = scaler.fit_transform(X_train)

        ## determine whether using predict or predict_proba
        prob = False if 'prob' not in kwargs else kwargs["prob"]
        
        if classifier == "SVM":
            ## setup a svm classifier
            kernel = "rbf" if 'kernel' not in kwargs else kwargs["kernel"]
            ## cost: default 1
            C = 1.0 if "C" not in kwargs else kwargs["C"]
            ## gamma: default (1/num_features)
            num_features = X_train.shape[1]
            gamma = (1.0/num_features) if "gamma" not in kwargs else kwargs["gamma"]

            self.clf = svm.SVC(C=C, gamma=gamma, kernel=kernel, probability=prob)

            self.params = "%s_%s" % (classifier, kernel)
        elif classifier == "SGD":

            shuffle = True if 'shuffle' not in kwargs else kwargs['shuffle']
            if prob:
                self.clf = SGDClassifier(loss="log", shuffle=shuffle)
            else:
                self.clf = SGDClassifier(shuffle=shuffle)

            self.params = "%s_%s" % (classifier, 'linear')
        elif classifier == "GaussianNB":
            self.clf = GaussianNB()

            self.params = "%s_%s" % (classifier, 'NB')
        else:
            raise Exception("currently only support SVM, SGD and GaussianNB classifiers")

        logging.debug('training with %s classifier' % (classifier))
        self.clf.fit(X_train, y_train)

    def predict(self, prob=True):

        if prob:
            self.predict_results = self.clf.predict_proba(self.X)
        else:
            self.predict_results = self.clf.predict(self.X)
        return self.predict_results


    def save_model(self, root=".", feature_name="", ext=".model"):
        
        if not self.feature_name:
            if feature_name:
                self.feature_name = feature_name
            else:
                logging.warn("speficy the feature_name for the file to be saved")
                return False
        out_path = os.path.join(root, self.feature_name+"."+self.params+".model" )
        out_dir = os.path.dirname(out_path)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        pickle.dump(self.clf, open(out_path, "wb"), protocol=2)
        logging.info("dump model to %s" %(out_path))

        return out_path

    def load_model(self, path):
        logging.info("loading model from %s" % (path))
        self.clf = pickle.load(open(path))

    def kFold(self, **kwargs):
        from sklearn import svm
        from sklearn.linear_model import SGDClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.cross_validation import KFold

        amend = False if "amend" not in kwargs else kwargs["amend"]
        if amend:
            ## amend dense matrix: replace NaN and None with float values
            self.check_and_amend()
        else:
            logging.debug("skip the amending process")

        # config n-fold verification
        n_folds = 10 if 'n_folds' not in kwargs else kwargs['n_folds']
        shuffle = True if 'shuffle' not in kwargs else kwargs['shuffle']

        ## get #(rows) of self.X
        n = utils.getArrayN(self.X)

        ## setup a kFolder
        kf = KFold(n=n , n_folds=n_folds, shuffle=shuffle )
        logging.debug("setup a kFold with n=%d, n_folds=%d" % (n, n_folds))
        
        ## setup a Scaler
        logging.debug("setup a StandardScaler")
        with_mean = True if 'with_mean' not in kwargs else kwargs['with_mean']
        with_std = True if 'with_std' not in kwargs else kwargs['with_std']

        ## setup a classifier
        classifier = "SVM" if "classifier" not in kwargs else kwargs["classifier"].upper()

        ## setup a svm classifier
        kernel = "rbf" if 'kernel' not in kwargs else kwargs["kernel"]

        # Cannot center sparse matrices, `with_mean` should be set as `False`
        if utils.isSparse(self.X):
            with_mean = False
        

        scaler = StandardScaler(with_mean=with_mean, with_std=with_std)

        logging.debug('training with %s classifier' % (classifier))


        ## determine whether using predict or predict_proba
        prob = False if 'prob' not in kwargs else kwargs["prob"]


        for (i, (train_index, test_index)) in enumerate(kf):

            logging.debug("train: %d , test: %d" % (len(train_index), len(test_index)))

            logging.info('cross validation round %d' % (i+1))
            X_train, X_test, y_train, y_test = self.X[train_index], self.X[test_index], self.y[train_index], self.y[test_index]


            logging.debug("scaling")

            X_train = scaler.fit_transform(X_train)
            X_test = scaler.fit_transform(X_test)

            if classifier == "SVM":
                clf = svm.SVC(kernel=kernel, probability=prob)
            elif classifier == "SGD":
                if prob:
                    clf = SGDClassifier(loss="log")
                else:
                    clf = SGDClassifier()
            else:
                logging.error("currently only support SVM and SGD classifiers")
                return False
            

            logging.debug("training (#x: %d, #y: %d)" % (len(X_train), len(y_train)))
            clf.fit(X_train, y_train)

            
            score = clf.score(X_test, y_test)
            logging.debug('get score %.3f' % (score))

            
            if prob:
                logging.debug("predicting (#x: %d, #y: %d) with prob" % (len(X_test), len(y_test)))
                result = clf.predict_proba(X_test)
            else:
                logging.debug("predicting (#x: %d, #y: %d)" % (len(X_test), len(y_test)))
                result = clf.predict(X_test)

            self.kfold_results.append( (i+1, y_test, result, score, clf.classes_) )

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
                logging.warn("speficy the feature_name for the file to be saved")
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
                logging.warn("speficy the feature_name for the file to be saved")
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
