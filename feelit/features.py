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

import logging, os
from feelit import utils
import numpy as np

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

        ## specify the data_range
        >> lf.loads(root="/Users/Maxis/projects/emotion-detection-modules/dev/image/emotion_imgs_threshold_1x1_rbg_out_amend/out_f1", data_range=800)

        ## amend value, ie., None -> 0, "NaN" -> 0
        >> lf.loads(root="/Users/Maxis/projects/emotion-detection-modules/dev/image/emotion_imgs_threshold_1x1_rbg_out_amend/out_f1", data_range=800, amend=True)

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
        >> fm.fetch_transform('TFIDF', '53a1921a3681df411cdf9f38', data_range=800)
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

        logging.info("fetching documents from %s" % (collection_name))
        
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

            logging.debug('loading X...')
            self.Xs[fn] = data['X']
            logging.debug('loading y...')
            self.ys[fn] = data['y']

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
            self.X = np.concatenate(self.Xs.values(), axis=1)
            if reduce_memory: del self.Xs

            logging.debug('fusing y')
            self.y = self.ys[ self.ys.keys()[0] ]
            return True

    def dump(self, path="auto", ext=".npz"):

        ## text_DepPairs_LSA512.Xy
        ## '_'.join(x.split('.Xy')[0].split('_')[1:]) --> DepPairs_LSA512
        if path == "auto":
            path = '+'.join(sorted([ '_'.join(x.split('.Xy')[0].split('_')[1:]) for x in self.Xs.keys() ]))

        ## amend path
        path = path if not ext or path.endswith(ext) else path+ext
        logging.debug("dumping X, y to %s" % (path))
        np.savez_compressed(path, X=self.X, y=self.y)

class Learning(object):
    """
    usage:
        >> from feelit.features import Learning
        >> l = Learning(verbose=True)
        >> l.load(path="data/image_rgba_gist.Xy.npz")
        >> l.kFold(classifier="SVM")
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
        self.feature_name = path.split('/')[-1].replace('.Xy','').split('.npz')[0]
        self.fn = self.feature_name

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
        
    def kFold(self, **kwargs):

        from sklearn import svm
        from sklearn.linear_model import SGDClassifier

        from sklearn.cross_validation import KFold
        from sklearn.preprocessing import StandardScaler

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

        for (i, (train_index, test_index)) in enumerate(kf):

            logging.debug("train: %d , test: %d" % (len(train_index), len(test_index)))

            logging.info('cross validation round %d' % (i+1))
            X_train, X_test, y_train, y_test = self.X[train_index], self.X[test_index], self.y[train_index], self.y[test_index]


            logging.debug("scaling")

            X_train = scaler.fit_transform(X_train)
            X_test = scaler.fit_transform(X_test)

            if classifier == "SVM":
                clf = svm.SVC(kernel=kernel)
            elif classifier == "SGD":
                clf = SGDClassifier()
            else:
                logging.error("currently only support SVM and SGD classifiers")
                return False
            

            logging.debug("training (#x: %d, #y: %d)" % (len(X_train), len(y_train)))
            clf.fit(X_train, y_train)

            
            score = clf.score(X_test, y_test)
            logging.debug('get score %.3f' % (score))

            logging.debug("predicting (#x: %d, #y: %d)" % (len(X_test), len(y_test)))
            result = clf.predict(X_test)

            self.kfold_results.append( (i+1, y_test, result, score) )

    def save(self, root=".", feature_name="", ext=".npz"):
        if not self.feature_name:
            if feature_name:
                self.feature_name = feature_name
            else:
                logging.warn("speficy the feature_name for the file to be saved")
                return False

        tests, predicts, scores = [], [], []
        for i, y_test, result, score in self.kfold_results:
            tests.append( y_test )
            predicts.append( result )
            scores.append( score )


        out_path = os.path.join(root, self.feature_name+".res"+ext )
        
        if not os.path.exists(os.path.dirname(out_path)): os.makedirs(os.path.dirname(out_path))

        np.savez_compressed(out_path, tests=tests, predicts=predicts, scores=scores)


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
