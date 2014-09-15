

import sys, os
sys.path.append("../")
from feelit.features import LIBSVM
from feelit import utils
classifier = "LIBSVM"   ## static value, don't modify it
classtype = "binary"    ## static value, don't modify it
emotions = utils.LJ40K

kernel = "rbf"
prob = True
## params
gamma = "default" # str(<float>) or "default"
C = 2.0

if __name__ == '__main__':       
    
    key = ["classifier", "kernel", "classtype", "prob"]
    val = [classifier, kernel, classtype, prob]

    arg = '_'.join(map(lambda a: '='.join([a[0], str(a[1])]), sorted(zip(key, val), key=lambda x:x[0])))
    
    _g = gamma
    _c = C

    # e.g., -t 0 -c 4 -b 1 -g 0.001 -q
    if kernel == "linear":
        _t = 0
    elif kernel == "polynomial":
        _t = 1
    elif kernel == "rbf":
        _t = 2
    elif kernel == "sigmoid":
        _t = 3
    else:
        print 'unknown kernel type'
        exit(-1)

    ## set probability
    _b = 1 if prob else 0

    if _g != "default":
        param = "-t %d -c %d -b %d -g %s -q" % ( _t, _c, _b, _g )
    else:
        param = "-t %d -c %d -b %d -q" % ( _t, _c, _b )

    print 'svm params:', param
    print zip(key, val)
    
    ## output root folder
    ## <feature_name>.classifier=<classifier>_kernel=<kernel>_classtype=<classtype>
    ## image_rgb_gist.classifier=SGD_kernel=linear_classtype=binary

    if len(sys.argv) >= 2:
    
        feature_name = sys.argv[1]
        
        if len(sys.argv) == 4:
            _from = int(sys.argv[2])
            _to = int(sys.argv[3])
            emotions = emotions[_from:_to]
    else:
        print 'usage: python %s <feature_name> [from, to]' % (__file__)
        exit()
    
    print 'will process', len(emotions), 'emotions: ', emotions, 'go?', raw_input()
    
    out_root = os.path.join("../models/", feature_name+'.'+arg+".models")

    for emotion in emotions:

        ## check if existed
        if os.path.exists(os.path.join(out_root,feature_name+"."+emotion)):
            print '>>> skip', emotion
            continue

        print '>>> processing', emotion

        src_path = "../train/"+feature_name+"/Xy/"+feature_name+".Xy."+emotion+".train.npz"

        print '>> loading', src_path
        svm = LIBSVM(verbose=True)
        svm.load_train(src_path)

        print '>> set param:',param
        svm.set_param(param)

        print '>> formulate the problem'
        svm.formulate()

        print '>> training'
        svm.train()

        print '>> saving'
        svm.save_model(root=out_root)
         
