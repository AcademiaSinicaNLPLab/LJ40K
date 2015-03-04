## Get and Print npz file shape

import sys, os
sys.path.append("../")
import numpy as np
from feelit import utils

def help():
    print "usage: python [file_path]"
    print
    print "  e.g: python ../exp/data/from_mongo/TFIDF+keyword_eachfromMongo.Xy.train.npz"
    exit(-1)

if __name__ == '__main__':
	
    if len(sys.argv) != 2: help()

    try:
        data = np.load(sys.argv[1])    
    except:
        print "Failed to load file %s" % (sys.argv[1])
        exit(-1)

    matrix_files = data.files
    print "File \"%s\" includes" % (sys.argv[1])
    print (matrix_files)
    print 
   
    for matrix_name in matrix_files:
 
        x = data[matrix_name]
        if utils.isSparse(x):
            print ("the npz file you check is a sparse matrix")
            x = utils.toDense(x)
        ## print matrix x
        print type(x), "\"%s\" = " % (matrix_name)
        print "%s" % (x)
        print   
        ## print shape
        print "Its dimension  is" 
        print (x.shape)
        print "--------------------------------------------"
        print

