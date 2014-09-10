## split data

import sys, os
sys.path.append("../")
from feelit.features import Learning
from feelit import utils

emotions = utils.LJ40K

if __name__ == '__main__':
    
    if len(sys.argv) == 3:
        _from = int(sys.argv[1])
        _to = int(sys.argv[2])
        emotions = emotions[_from:_to]

    print 'will process', len(emotions), 'emotions: ', emotions, 'go?', raw_input()


    feature_name = "text_TFIDF"

    out_root = "../results/text_TFIDF_binary"

    

    for emotion in emotions[:5]:

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
        l.kFold(classifier="SVM", kernel="linear")

        print '>> saving'
        l.save(root=out_root)










         