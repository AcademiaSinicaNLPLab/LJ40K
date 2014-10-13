
import sys
sys.path.append("../")
from feelit.kernel import RBF
import os
import numpy as np

rbf = RBF(verbose=True)

in_path = "../data/text_TFIDF/Xy/text_TFIDF.Xy."+sys.argv[1]+".npz"
print 'loading', in_path
rbf.load( in_path )

print 'building'
##### for debuging #####
# rbf._test_build()
rbf.build()

## check output dir
root = "../data/text_TFIDF/kernels/"
if not os.path.exists(root): os.makedirs(root)

out_fn = "text_TFIDF.K."+sys.argv[1]+".npz"
out_path = os.path.join( root, out_fn )
print 'dumping', out_path
np.savez_compressed(out_path, K=rbf.Ksmall)
