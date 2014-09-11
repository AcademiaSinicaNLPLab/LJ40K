## split data

import sys, os
sys.path.append("../")
from feelit.features import Learning
from feelit import utils

emotions = utils.LJ40K

if __name__ == '__main__':

    classifier = "SVM"
    kernel = "rbf"
    classtype = "binary"
    prob = True

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
    
    out_root = os.path.join("../results/", feature_name+'.'+arg)

    for emotion in emotions:

        ## check if existed
        if os.path.exists(os.path.join(out_root,feature_name+"."+emotion)):
            print '>>> skip', emotion
            continue

        print '>>> processing', emotion

        src_path = "../data/"+feature_name+"/Xy/"+feature_name+".Xy."+emotion+".npz"

        print '>> loading', src_path
        l = Learning(verbose=True)
        l.load(path=src_path)

        print '>> training'
        l.kFold(classifier=classifier, kernel=kernel, prob=prob)

        print '>> saving'
        l.save(root=out_root)
         
