# -*- coding: utf-8 -*-

##########################################
# classes:
#   feelit > kernel > RBF
#
# function: 
#   compute kernel, transform formats
#
#   -MaxisKao @ 20140904
##########################################

import logging, os
from feelit import utils
import numpy as np
from math import exp

class RBF(object):
    """
    build RBF kernel matrix from the input feature vectors

    usage:
        >> from feelit.kernel import RBF
        >> rbf = RBF(verbose=True)
        >> rbf.load(path="text_TFIDF.Xy.npz")
        
    maybe do random sampling before build:
        >> from feelit.utils import RandomSample

        # keep only 10%
        >> rbf.X, rbf.y = RandomSample((rbf.X, rbf.y), 0.1)

        # use certain index file if `idxs.pkl` exists
        >> rbf.X, rbf.y = RandomSample((rbf.X, rbf.y), index_file="idxs.pkl")

        # save the index into `index_file` if `new_idxs.pkl` didn't exist
        >> rbf.X, rbf.y = RandomSample((rbf.X, rbf.y), 0.1, index_file="new_idxs.pkl")

    and devide into training and developing sets

        >> from feelit.utils import devide
        >> tr, dev = devide(rbf.X, 0.9) # 90% - 10%

    build matrices `K_tr` and `K_dev`

        >> K_tr = rbf.build(tr, tr)
        >> K_dev = rbf.build(tr, dev)

    save results

        >> rbf.dump(path="data/text_TFIDF.Ky.npz")

        or also save csv
        >> rbf.dump(path="data/text_TFIDF.Ky.npz", toCSV=True)
    """
    def __init__(self, **kwargs):
        """
        Parameters
        ==========
            verbose: True/False
        """        
        loglevel = logging.DEBUG if 'verbose' in kwargs and kwargs['verbose'] == True else logging.INFO
        logging.basicConfig(format='[%(levelname)s] %(message)s', level=loglevel)        
        
        self.X = None
        self.y = None
        self.K = None
        
    def load(self, path):
        """
        load (X,y) from a .npz file

        Parameters
        ==========
            path: path to the .npz file storing feature vectors
        Returns:
            X, y
        """
        data = np.load(path)

        if 'X' not in data.files or 'y' not in data.files:
            logging.error("check the format in %s, must have X and y inside." % (path))
            return False

        ## sparse: if not data['X'].shape
        ## dense:  if data['X'].shape
        self.X = data['X'] if data['X'].shape else data['X'].all().toarray()
        self.y = data['y']

        self._shape_X_str = 'x'.join(map(lambda x:str(x), self.X.shape))
        self._shape_y_str = 'x'.join(map(lambda x:str(x), self.y.shape))

        logging.info("X: %s and y: %s have been loaded" % (self._shape_X_str, self._shape_y_str))

    def _squared_Euclidean_distance(self, p, q):
        return sum([ (_p-_q)*(_p-_q) for _p, _q in zip(p, q)])

    def _rbf_kernel_function(self, v1, v2, gamma="default"):
        if gamma == "default":
            num_features = len(v1)
            gamma = 1.0/num_features

        sed = self._squared_Euclidean_distance(v1, v2)
        k_v1_v2 = exp(-1.0*gamma*sed)
        return k_v1_v2

    def _test_build(self):
        ## build a 10x10 matrix for debug
        self.Ksmall = np.zeros((10, 10))
        for i in range(10):
            for j in range(10):
                self.Ksmall[i][j] = self.Ksmall[j][i] = self._rbf_kernel_function(self.X[i], self.X[j], gamma="default")

    def build(self, A, B):
        """
        Parameters
        ==========

        """
        # determine shape
        m, n = len(A), len(B)
        self.K = np.zeros((m, n))

        logging.info("building K from (%s) and (%s)" % (utils.strShape(A), utils.strShape(B)))

        for i in xrange(m):
            for j in xrange(n):
                self.K[i][j] = self._rbf_kernel_function(A[i], B[j], gamma="default")

        logging.info("K with size %s has been built." % utils.strShape(self.K))

    def _build(self, X=None):
        """
        Parameters
        ==========
            X: numpy.ndarray (optional)
                n row: n documents
                m col: m features

            X = [ v_1, ..., v_n ], where

            v_1: [f1, ..., fm]
            v_2: [f1, ..., fm]
               ...
            v_n: [f1, ..., fm]

        Example
        =======
        X : <32000x15456>, which contains 32,000 documents. Each doc contains 15,456 features
        """ 

        if X != None: 
            self.X = X

        _num_of_samples = len(self.X)
        logging.debug( 'num of samples: %d' % (_num_of_samples) )

        logging.debug( 'build zero matrix' )
        self.K = np.zeros((_num_of_samples, _num_of_samples))

        logging.debug( 'build the rbf-matrix `self.K`' )
        for i in range(_num_of_samples):
            self.K[i][i] = 1.0
            for j in range(i+1, _num_of_samples):
                self.K[i][j] = self.K[j][i] = self._rbf_kernel_function(self.X[i], self.X[j], gamma="default")

    def dump(self, path, toCSV=False):
        """
        Parameters
        ==========
        path: str
            path to the matrix containing `K` and `y`

        Example
        =======
        >> rbf.dump(path="data/text_TFIDF.Ky.npz", toCSV=True)

        """
        logging.debug("dumping K and y to %s" % (path))
        np.savez_compressed(path, K=self.K, y=self.y)
        
        if toCSV:
            csv_path = '.'.join(path.split('.')[:-1]+['csv'])
            logging.info('save csv version in %s' % (csv_path))
            np.savetxt(csv_path, self.K, delimiter=",")
