# npz to csv
import numpy as np
import os

feature = 'rgba_gist+rgba_phog'

Ky_root = 'exp/train/'+feature+'/Ky/'
csv_root = 'exp/train/'+feature+'/csv/'

classcode = [1, -1]

if not os.path.exists(csv_root): os.makedirs(csv_root)
npzs = filter(lambda x:x.endswith('.npz'), os.listdir(Ky_root))

for npz in npzs:

    # "rgba_gist+rgba_phog.Ky.happy.dev.npz"
    # [feature].[content].[emotion].[dtype].[ext]

    npz_path = os.path.join(Ky_root, npz) 

    data = np.load(npz_path)

    feature, content, emotion, dtype, ext = npz.split('.') 

    if npz.endswith('.dev.npz'):
        dtype = 'dev'
    elif  npz.endswith('.train.npz'):
        dtype = 'train'
    else:
        continue

    K = data['K_'+dtype]
    y = [classcode[1] if x.startswith('_') else classcode[0] for x in data['y_'+dtype]]

    if dtype == 'dev':
        # to csv
    np.savetxt(os.path.join(csv_root, feature+'.K.'+emotion+'.'+dtype+'.csv'),   K,   delimiter=",", fmt="%.6f")
    np.savetxt(os.path.join(csv_root, feature+'.y.'+emotion+'.'+dtype+'.csv'),   y,   delimiter=",", fmt="%d")
