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
        >> rbf = RBF()
        >> rbf.load(path="data/text_TFIDF.Xy.npz")
        
    maybe do random sampling before build:
        >> from feelit.utils import RandomSample
        >> rbf.X, rbf.y = RandomSample((rbf.X, rbf.y), 0.1) # keep only 10%
        >> rbf.X, rbf.y = RandomSample((rbf.X, rbf.y), index_file="data/idxs.pkl") # use certain index file

    build the matrix K
        >> >> rbf.build()
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

    def _squared_Euclidean_distance(self, p, q):
        return sum([ (_p-_q)*(_p-_q) for _p, _q in zip(p, q)])

    def _rbf_kernel_function(self, v1, v2, gamma="default"):
        if gamma == "default":
            num_features = len(v1)
            gamma = 1.0/num_features

        sed = self._squared_Euclidean_distance(v1, v2)
        k_v1_v2 = exp(-1.0*gamma*sed)
        return k_v1_v2 

    def build(self, X=None):
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

        if X: self.X = X

        _num_of_samples = len(self.X)

        self.K = np.zeros((_num_of_samples, _num_of_samples))
        for i in range(_num_of_samples):
            self.K[i][i] = 1.0
            for j in range(i+1, _num_of_samples):
                self.K[i][j] = self.K[j][i] = self._rbf_kernel_function(self.X[i], self.X[j], gamma="default")


