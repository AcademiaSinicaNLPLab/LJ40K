# -*- coding: utf-8 -*-

import numpy as np
import pickle
import pymongo
import logging

from collections import defaultdict

from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import KFold
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.decomposition import PCA, SparsePCA


class FeatureFusion(object):
    """
    A scikit-learn wrapper: fuse several features from mongodb
    """
    def __init__(self, **kwargs):

        loglevel = logging.DEBUG if 'verbose' in kwargs and kwargs['verbose'] == True else logging.INFO
        logging.basicConfig(format='[%(levelname)s] %(message)s', level=loglevel)

        self.feature_names = set()

        self._feature = []
        self._target = []

        self.Xs = []
        self.ys = []

        self.kfold_results = []

        self._db = None

    def connect(self, **kwargs):
        ## config mongodb
        mongo_addr = 'doraemon.iis.sinica.edu.tw' if 'mongo_addr' not in kwargs else kwargs['mongo_addr']
        db_name = 'LJ40K' if 'db_name' not in kwargs else kwargs['db_name']

        self._db = pymongo.Connection(mongo_addr)[db_name]

        self.emoID_map = {e:i for i, e in enumerate(sorted(map(lambda x:x['emotion'], self._db["emotions"].find({ 'label': db_name }))))}

    def _getCollectionName(self, feature_name, prefix="features"):
        return '.'.join([prefix, feature_name])

    def _toNumber(string, NaN=-1):
        return NaN if string.lower() == 'nan' else float(string)

    



    ### mongodb
    def add_mongo(self, feature_name):
        """
        feature_name    :   e.g., TFIDF, pattern_emotion, keyword, ... ,etc.
        """
        collection_name = self._getCollectionName(feature_name)

        if not self._db:
            self.connect()

        if collection_name not in self._db.collection_names():
            raise Exception('cannot find collection %s in %s' % (collection_name, self._db.name))
        else:
            if feature_name in self.feature_names:
                logging.info('feature %s already exists' % (feature_name))
            else:
                self.feature_names.add( feature_name )
                logging.info('feature %s added' % (feature_name))
        return self.feature_names

    ### mongodb
    def fuse_mongo(self, label_name="emotion", **kwargs):
        """
        !! could cause heavy memory usage !!
        """
        for feature_name in self.feature_names:

            ## setup collection name
            ## feature_name (e.g. TFIDF) --> collection_name (e.g. features.TFIDF)
            collection_name = self._getCollectionName(feature_name)
            co = self._db[collection_name]


            logging.info('extracting %s features from %s' % (feature_name, co.full_name) )

            ## start to fetch mongodb
            _count = co.count()
            batch_size = _count/25 if 'batch_size' not in kwargs else kwargs['batch_size']

            for i, mdoc in enumerate(co.find().batch_size(batch_size)):

                if type(mdoc['feature']) == dict: pass
                ## transform to dictionary
                elif type(mdoc['feature']) == list:
                    mdoc['feature'] = { f[0]:f[1] for f in mdoc['feature'] }
                else:
                    raise TypeError('make sure the feature format is either <dict> or <list> in mongodb')

                ### get target label
                label = mdoc[label_name]

                ## store all features in a list
                self._feature.append( mdoc['feature'] )
                self._target.append( label )

                logging.debug('mongo doc %d/%d fetched' % ( i+1, _count))

        ## vectorize
        logging.debug('vectorizing')
        vec = DictVectorizer()
        self.X = vec.fit_transform( self._feature )
        self.y = np.array( self._target )

    ### load from file, one line one feature
    def add_file(self, path, label="$filename", LINE="\n", ITEM=","):
        """
        one line one feature
        label: 
            "$filename"   : extract label from filename
                            e.g., accomplished_gist.csv --> accomplished
        """

        logging.debug('loading %s' % (path))

        doc = open(path).read()
        lines = doc.strip().split(LINE)

        ## one line one feature --> one line one document
        lines_num = np.array( [ map(lambda x:_toNumber(x), line.split(ITEM)) for line in lines] )
        lines_num_T = lines_num.transpose()

        # _samples = defaultdict(list)
        # for fi, line in enumerate(lines): # i-th feature (680 lines --> 680 feautres)
        #     samples = line.strip().split(ITEM)
        #     for si, sample in enumerate(samples): # i-th sample (995 articles --> 995 samples)
        #         feature_value = self._toNumber(sample)
        #         _samples[si].append(feature_value)

        X = []
        for si in _samples:
            X.append(_samples[si])

        ## assign label to this loaded data
        _label = label if not label == "$filename" else path.split('/')[-1].split('.')[0].split('_')[0]

        y = [_label]*len(X)

        self.Xs += X
        self.ys += y

    def nFold(self, **kwargs):
        """
        alias for kFold() function
        """
        self.kFold(**kwargs)

    def kFold(self, **kwargs):
        """
        kwargs:
            n_folds :   default 10
            shuffle :   True
        """

        # config n-fold verification
        n_folds = 10 if 'n_folds' not in kwargs else kwargs['n_folds']
        shuffle = True if 'shuffle' not in kwargs else kwargs['shuffle']
        kf = KFold( len(self._feature), n_folds=n_folds, shuffle=shuffle )

        loss = 'modified_huber' if 'loss' not in kwargs else kwargs['loss']
        penalty = 'elasticnet' if 'penalty' not in kwargs else kwargs['penalty']

        ## setup a Scaler
        with_mean = False if 'with_mean' not in kwargs else kwargs['with_mean']
        scaler = StandardScaler(with_mean=False)

        for (i, (train_index, test_index)) in enumerate(kf):
            logging.info('cross validation round %d' % (i+1))

            X_train, X_test, y_train, y_test = self.X[train_index], self.X[test_index], self.y[train_index], self.y[test_index]
            
            #scale data for SGD performance
            logging.debug('scaling')
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
            
            ## train
            # logging.info('training with SGDClassifier')
            # clf = SGDClassifier(loss=loss, penalty=penalty, shuffle=shuffle)

            logging.info('training with svm.SVC classifier')

            clf = svm.SVC()
            ## libsvm
            ## cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
            ## gamma : set gamma in kernel function (default 1/num_features)
            ## epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)
            ## cachesize : set cache memory size in MB (default 100)
            ## tolerance : set tolerance of termination criterion (default 0.001)
            ## shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)
            ## probability_estimates : whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)

            # SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
            #     gamma=0.0, kernel='rbf', max_iter=-1, probability=False, random_state=None,
            #     shrinking=True, tol=0.001, verbose=False)
            clf.fit(X_train, y_train)

            ## scoring
            score = clf.score(X_test, y_test)
            logging.info('score %.3f' % (score))

            ## predict
            logging.info('predicting')
            result = clf.predict(X_test)

            self.kfold_results.append( (i+1, y_test, result) )

    def save(self, root="."):

        ## generate fused feature name
        fused_feature_name = '+'.join(sorted(list(self.feature_names)))

        for ith, y_test, result in self.kfold_results:

            out_fn = "%s.fold-%d.result" % (fused_feature_name, ith)

            with open(out_fn, 'w') as fw:

                fw.write( ' '.join( map(lambda x:str(x), y_test) ) )
                fw.write('\n')

                fw.write( ' '.join( map(lambda x:str(x), result) ) )
                fw.write('\n')
        
if __name__ == '__main__':

    ## %load_ext autoreload
    ## %autoreload 2

    from FeatureFusion import FeatureFusion

    ff = FeatureFusion(verbose=True)
    ff.add_mongo(feature_name="TFIDF")
    ff.fuse_mongo()


    # spca = SparsePCA(n_components=0.5)
    # spca.fit_transform(ff.X)

    # X = ff.X.toarray()
    # pca = PCA(n_components=2)
    # pca.fit_transform(X)


    # F = FeatureFusion(mongo_addr="doraemon.iis.sinica.edu.tw", db_name="LJ40K", verbose=True)

    # F.add(feature_name="TFIDF")
    # F.add(path="/Users/Maxis/projects/emotion-detection-modules/dev/image/emotion_imgs_threshold_1x1_rbg_out/out_f1/accomplished_gist.csv", label="accomplished")
    # F.add(path="/Users/Maxis/projects/emotion-detection-modules/dev/image/emotion_imgs_threshold_1x1_rbg_out/out_f1/sleepy_gist.csv", label="sleepy")

    # F.fuse_mongo()

    # F.kFold(n_folds=10, shuffle=True, loss="modified_huber", penalty="elasticnet")

    # F.save()

            

