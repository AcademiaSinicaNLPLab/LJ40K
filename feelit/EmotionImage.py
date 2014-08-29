
from collections import defaultdict

import numpy as np
import os
import logging

## feature extraction
from sklearn.feature_extraction import DictVectorizer

## classifiers
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC

from sklearn.cross_validation import KFold
from sklearn.preprocessing import StandardScaler



class EmotionImage(object):
    """
    EmotionImage
    A scikit-learn wrapper for training Emotion-Image
    """
    __support_classifiers__ = ['SGD', 'SVC']

    def __init__(self, **kwargs):
        
        loglevel = logging.DEBUG if 'verbose' in kwargs and kwargs['verbose'] == True else logging.INFO
        logging.basicConfig(format='[%(levelname)s] %(message)s', level=loglevel)    

        self.features = defaultdict()
        self.Xs = {}
        self.ys = {}
        self.kfold_results = []
        # self.sample_num = -1
    
    def _toNumber(self, string, NaN=-1):
        return NaN if string.lower() == 'nan' else float(string)

    def load(self, path, label="filename", LINE="\n", ITEM=","):
        """
        one line one feature

        """
        logging.debug('loading %s' % (path))

        doc = open(path).read()
        lines = doc.strip().split(LINE)

        _samples = defaultdict(list)
        for fi, line in enumerate(lines): # i-th feature (680 lines --> 680 feautres)
            samples = line.strip().split(ITEM)
            for si, sample in enumerate(samples): # i-th sample (995 articles --> 995 samples)
                feature_value = self._toNumber(sample)
                _samples[si].append(feature_value)

        X = []
        for si in _samples:
            X.append(_samples[si])

        ## assign label to this loaded data
        _label = label if not label == "$filename" else path.split('/')[-1].split('.')[0].split('_')[0]

        y = [_label]*len(X)

        self.Xs[_label] = X
        self.ys[_label] = y

    def loads(self, root, LINE="\n", ITEM=",", ext=None):
        for fn in os.listdir(root):
            if ext and not fn.endswith(ext):
                continue
            else:
                self.load( path=os.path.join(root, fn), LINE=LINE, ITEM=ITEM )

    def _assembly(self, Xs, ys):
        """
        collect all loaded Xs, ys
        """
        self.X = []
        self.y = []
        for label in self.Xs:
            self.X += self.Xs[label]
            self.y += self.ys[label]

        self.X = np.array(self.X)
        self.y = np.array(self.y)

    def run(self, **kwargs):

        ## transform Xs and ys to numpy array before setting up the KFlod
        self._assembly()

        ## config n-fold verification
        ## run _assembly() first
        n_folds = 10 if 'n_folds' not in kwargs else kwargs['n_folds']
        shuffle = True if 'shuffle' not in kwargs else kwargs['shuffle']
        kf = KFold( len(self.X), n_folds=n_folds, shuffle=shuffle )


        ## setup a Scaler
        with_mean = False if 'with_mean' not in kwargs else kwargs['with_mean']
        scaler = StandardScaler(with_mean=False)

        ## SGD
        loss = 'modified_huber' if 'loss' not in kwargs else kwargs['loss']
        penalty = 'elasticnet' if 'penalty' not in kwargs else kwargs['penalty']

        ## SVC
        C = 1.0 if 'C' not in kwargs else kwargs['C']
        gamma = 0.0 if 'gamma' not in kwargs else kwargs['gamma']
        probability = False if 'probability' not in kwargs else kwargs['probability']
        shrinking = True if 'shrinking' not in kwargs else kwargs['shrinking']

        ## set classifier
        classifier = self.__support_classifiers__[0] if 'classifier' not in kwargs else kwargs['classifier']
        if classifier not in self.__support_classifiers__:
            raise Exception('unkown classifier, currently supports %s' % ', '.join(self.__support_classifiers__) )

        for (i, (train_index, test_index)) in enumerate(kf):

            logging.info('cross validation round %d' % (i+1))

            X_train, X_test, y_train, y_test = self.X[train_index], self.X[test_index], self.y[train_index], self.y[test_index]

            #scale data for SGD performance
            logging.debug('scaling')
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)            

            if classifier == 'SGD':
                logging.info('training with SGDClassifier')
                clf = SGDClassifier(loss=loss, penalty=penalty, shuffle=shuffle)

            elif classifier == 'SVC':
                logging.info('training with svm.SVC')
                clf = SVC(C=C, gamma=gamma, probability=probability, shrinking=shrinking)
            else:
                raise Exception('unkown classifier')

            clf.fit(X_train, y_train)

            ## scoring
            score = clf.score(X_test, y_test)
            logging.info('score %.3f' % (score))

            ## predict
            logging.info('predicting')
            result = clf.predict(X_test)

            self.kfold_results.append( (i+1, y_test, result) )

    def save(self, feature_name="$fusion", root="."):

        if not kfold_results:
            logging.warn('Nothing to be saved!')
            return False

        if not os.path.exists(root):
            os.makedirs(root)

        _feature_name = '+'.join(sorted(list(self.feature_names))) if feature_name == "$fusion" else feature_name

        for ith, y_test, result in self.kfold_results:

            out_fn = "%s.fold-%d.result" % (_feature_name, ith)

            with open( os.path.join(root, out_fn), 'w' ) as fw:

                fw.write( ','.join( map(lambda x:str(x), y_test) ) )
                fw.write('\n')

                fw.write( ','.join( map(lambda x:str(x), result) ) )
                fw.write('\n')

if __name__ == '__main__':

    # from EmotionImage import EmotionImage

    Eimg = EmotionImage(verbose=True)

    # Eimg.loads('fusion_rgb', ext="csv")
    Eimg.loads('emotion_imgs_threshold_1x1_rbga_out/out_f1', ext="csv")

    # Eimg.run(classifier="SGD", n_folds=10, shuffle=True, loss="modified_huber", penalty="elasticnet")
    Eimg.run(classifier="SVC", n_folds=10, C=4.0)

    Eimg.save(feature_name="fusion_rgb", root="results/fusion_rgb_SVC")
