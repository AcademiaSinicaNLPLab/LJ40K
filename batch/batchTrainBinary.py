## split data

import sys, os
sys.path.append("../")
from feelit.features import Learning
from feelit import utils
classtype = "binary"    ## static value, don't modify it
emotions = utils.LJ40K  

classifier = "SVM"
kernel = "rbf"
prob = True
## params
gamma = "default" # str(<float>) or "default"
C = 2.0

if __name__ == '__main__':



    key = ["classifier", "kernel", "classtype", "prob"]
    val = [classifier, kernel, classtype, prob]

    arg = '_'.join(map(lambda a: '='.join([a[0], str(a[1])]), sorted(zip(key, val), key=lambda x:x[0])))


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
        l = Learning(verbose=True)
        l.load(path=src_path)

        print '>> training'
        if gamma != "default":
            l.train(classifier=classifier, kernel=kernel, prob=prob, gamma=float(gamma), C=float(C) )
        else:
            l.train(classifier=classifier, kernel=kernel, prob=prob, C=float(C) )

        print '>> saving'
        l.save_model(root=out_root)
         
