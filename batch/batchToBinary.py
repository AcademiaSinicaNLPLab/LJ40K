## split data

import sys
sys.path.append("../")
import numpy as np
from feelit import utils
import pickle
from feelit.features import dump


def gen_idx(y):
    # Return: 
    # {
    #     <label>: [positive_ins, negative_ins],
    #     ...
    # }
    from collections import Counter
    import random

    dist = Counter(y)

    G = {}

    for label in dist:
        # filter out

        possitive_samples, negative_candidates = [], []
        for i, _label in enumerate(y):
            if _label == label:
                possitive_samples.append( (i, _label) )
            else:
                negative_candidates.append( (i, _label) )

        negative_samples = random.sample(negative_candidates, len(possitive_samples))

        G[label] = possitive_samples + negative_samples

    return G

def subsample(X, y, idxs):
    """
    subsample a 2-D array by row index
    """
    _X, _y = [], []
    for i in xrange(len(X)):
        if i in idxs:
            _y.append( y[i] )
            _X.append( X[i] )
    return ( np.array(_X), np.array(_y) )

def relabel(y, label):
    return [label if _y == label else "_"+label for _y in y ]


def save(G, path="random_idx.pkl"):
    pickle.dump(G, open(path, "wb"), protocol=2)

def load(path="random_idx.pkl"):
    return pickle.load(open(path))

if __name__ == '__main__':
    
    # feature_name = "image_rgba_gist"
    feature_name = "text_TFIDF"
    data = np.load("../data/"+feature_name+".Xy.npz")
    
    X = utils.toDense( data['X'] )
    # data['X']:
    ## array(<32000x2732 sparse matrix of type '<type 'numpy.float64'>'
    ##     with 914162 stored elements in Compressed Sparse Row format>, dtype=object)

    y = data['y']
    # array([u'accomplished', u'accomplished', u'accomplished', ..., u'tired',
    #        u'tired', u'tired'],
    #       dtype='<U13')

    ## generate
    # G = gen_idx(y)

    ## load from existed G
    G = load(path="random_idx.pkl")

    for i_label, label in enumerate(G):
        print 'processing %d/%d' % ( i_label+1, len(G) )
        print ' > subsampling', label
        idxs = set([i for i,l in G[label]])
        _X, _y = subsample(X, y, idxs)

        _y = relabel(_y, label)

        path = "../data/"+feature_name+"/Xy/"+feature_name+".Xy."+label+".npz"
        print ' > dumping', path
        dump(path, X=_X, y=_y)









         