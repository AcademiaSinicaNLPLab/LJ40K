# npz to csv
import numpy as np
import os, sys


if len(sys.argv) != 2:
    print 'python batchToCSV.py [feature_name]'
    print 
    print '  e.g., feature_name: rgba_gist+rgba_phog'
    exit(-1)
# feature = 'rgba_gist+rgba_phog'
feature = sys.argv[1]

Ky_root = '../exp/train/'+feature+'/Ky/'
csv_root = '../exp/train/'+feature+'/csv/'

classcode = [1, -1]

if not os.path.exists(csv_root): os.makedirs(csv_root)
npzs = filter(lambda x:x.endswith('.npz'), os.listdir(Ky_root))

for npz in npzs:

    # "rgba_gist+rgba_phog.Ky.happy.dev.npz"
    # [feature].[content].[emotion].[dtype].[ext]

    npz_path = os.path.join(Ky_root, npz) 

    print 'load', npz
    data = np.load(npz_path)

    feature, content, emotion, dtype_full, ext = npz.split('.')

    if npz.endswith('.dev.npz'):
        dtype = 'dev'
    elif npz.endswith('.train.npz'):
        dtype = 'tr'
    else:
        continue

    K = data['K_'+dtype]
    y = [classcode[1] if x.startswith('_') else classcode[0] for x in data['y_'+dtype]]

    K_path = os.path.join(csv_root, feature+'.K.'+emotion+'.'+dtype+'.csv')
    y_path = os.path.join(csv_root, feature+'.y.'+emotion+'.'+dtype+'.csv')

    print 'save K to', K_path
    np.savetxt(K_path,   K,   delimiter=",", fmt="%.6f")

    print 'save y to', y_path
    np.savetxt(y_path,   y,   delimiter=",", fmt="%d")
