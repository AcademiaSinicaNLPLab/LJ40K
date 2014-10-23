import numpy as np
import os
import sys
sys.path.append("../")
from feelit import utils
from sklearn.preprocessing import StandardScaler

SCALING = False

def savetxt(fn, (X, y), scaling=False):
    if scaling:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        Xy = zip(X_scaled, y)
    else:
        Xy = zip(X, y)

    with open(fn, "w") as fw:
        for _X, _y in Xy:
            line = [ "0" if '_' in _y else "1" ]
            line += ["%d:%.6f" % (i+1, x) for i, x in enumerate(_X)]
            str_line = ' '.join( line ) + '\n'
            fw.write( str_line )

if __name__ == '__main__':

    
    feature_names = sys.argv[1:]

    for feature_name in feature_names:

        input_root = "../train/"+feature_name+"/Xy/"
        output_root = input_root.replace("Xy", "libsvm")

        print '>> processing', feature_name
        print ' > input:', input_root
        print ' > output:', output_root
        # output_root = "../train/image_rgba_gist/libsvm/"
        
        for npz_file in filter(lambda x:x.endswith(".npz"), os.listdir(input_root)):

            input_fn =  os.path.join( input_root, npz_file )
            print ' > loading', input_fn
            data = np.load(input_fn)

            
            if not os.path.exists( output_root ): os.makedirs( output_root )
            out_fn = npz_file.replace(".npz", ".txt" if not SCALING else ".scaled.txt")
            out_path = os.path.join( output_root, out_fn )

            print ' > saving', out_path, 'with scaling' if SCALING else ''
            savetxt(out_path, (data['X'], data['y']), scaling=SCALING )
    