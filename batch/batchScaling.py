

from sklearn.preprocessing import StandardScaler
import os
import sys
sys.path.append( "../" )
from feelit import utils
import numpy as np

def batchScaling(in_root="raw", out_root="data", with_mean=True, with_std=True):

    Xy_files = filter(lambda x:x.endswith(".Xy.npz"), os.listdir(in_root))
    # Xy_files = ["image_rgb_gist.Xy.npz"]

    for Xy_file in Xy_files:

        in_path = os.path.join( in_root, Xy_file )
        out_path = os.path.join( out_root, Xy_file )

        print '> load %s' % ( in_path )

        data = np.load( in_path )
        
        ## detect sparse or dense
        _sparse = True if len(data['X'].shape) == 0 else False

        print '> scaling'
        if _sparse:
            ## Cannot center sparse matrices: pass `with_mean=False` instead.
            print '>> Sparse matrix detected. Use with_mean=False'
            scaler = StandardScaler(with_mean=False, with_std=with_std)
            X = scaler.fit_transform( data['X'].all() )
        else:
            scaler = StandardScaler(with_mean=with_mean, with_std=with_std)
            X = scaler.fit_transform( data['X'] )

        
        print '> compressing and dumping to %s' % (out_path)
        np.savez_compressed(out_path, X=X, y=data['y'])

        print '='*50


if __name__ == '__main__':

    batchScaling(in_root="../raw", out_root="../data", with_mean=True, with_std=True)
