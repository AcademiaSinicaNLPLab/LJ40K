
import os
import numpy as np
from sklearn.preprocessing import StandardScaler

def savetxt(fn, (X, y), target, scaling=False):
    if scaling:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        Xy = zip(X_scaled, y)
    else:
        Xy = zip(X, y)

    with open(fn, "w") as fw:
        for _X, _y in Xy:
            line = [ "0" if _y != target else "1" ]
            line += ["%d:%f" % (i+1, x) for i, x in enumerate(_X)]
            str_line = ' '.join( line ) + '\n'
            fw.write( str_line )

if __name__ == '__main__':
    
    # "image_rgba_gist.Xy.calm.train.scaled.txt"

    src = "../data/image_rgba_phog.Xy.test.npz"

    target = 'calm'

    import numpy as np

    data = np.load(src)

    out_path = "../test/image_rgba_phog/libsvm/image_rgba_phog.Xy."+target+".test.txt"

    print out_path

    out_root = os.path.dirname(out_path)
    if not os.path.exists( out_root ): os.makedirs( out_root )

    savetxt(out_path, (data['X'], data['y']), target, scaling=True)

