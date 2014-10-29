# -*- coding: utf-8 -*-

##########################################
# classes:
#   feelit > features > LIBSVM
#
#   -MaxisKao @ 20141029
##########################################

import logging, os, sys
import numpy as np
__LIBSVM__ = "/tools/libsvm/python"

class LIBSVM(object):
    """
    currently only supports binary version
    i.e., the label should be either `happy` or `_happy`

    Usage:

    from feelit.features import LIBSVM
    svm = LIBSVM(verbose=True)

    ####################################
    ## load training samples
    svm.load_train("data/text_TFIDF.Xy.train.npz")

    ## set param for training
    svm.set_param("-t 0 -c 4 -b 1 -q")

    ## formulate problem
    svm.formulate()

    ## training
    svm.train()

    ## save model
    svm.save_model(root="models", filename="auto")

    ####################################

    ## load_model

    ## load testing samples
    svm.load_test("data/image_rgb_phog.Xy.test.npz")

    ## set param for testing
    svm.set_param("-b 1 -q")

    ## testing
    svm.predict()


    """
    if not os.path.exists(__LIBSVM__):
        logging.error("can't find the LIBSVM in %s" % (__LIBSVM__))
    
    sys.path.append(__LIBSVM__)

    import svmutil

    def __init__(self, **kwargs):
        self.verbose = True if 'verbose' in kwargs and kwargs['verbose'] == True else False
        loglevel = logging.DEBUG if self.verbose else logging.INFO
        logging.basicConfig(format='[%(levelname)s] %(message)s', level=loglevel)

        self.y_train, self.X_train, self.y_test, self.X_test = None, None, None, None
        self.feature_name = None
        self.label_map = None

    # def build_numeric_label_map(self, y):
    #     logging.debug("building label numeric mapping")
    #     self.label_map = {_y:i for i,_y in enumerate(sorted(list(set(y))))}
    #     pickle.dump(self.label_map, open("numeric_label_map.pkl", "wb"))
    # def numerize(self, y, label_map=None):
    #     if not label_map and not self.label_map:
    #         self.build_numeric_label_map(y)
    #     if not label_map:
    #         label_map = self.label_map
    #     return [label_map[_y] for _y in y]

    def numerize(self, y):
        ############################## future work ##############################
        if len(set(y)) != 2:
            raise Exception("currently only supports binary classifier")
        #########################################################################

        return [0 if _y.startswith("_") else 1 for _y in y]
        
    def scale(self, X, with_mean=True, with_std=True):
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler(with_mean=with_mean, with_std=with_std)
        return scaler.fit_transform(X)

    def load_train(self, path, scaling=False):
        data = np.load(path)

        _scaling_msg_ = "and scaling " if scaling else ""
        logging.debug("transforming %s X to list" % (_scaling_msg_))
        self.X_train = data['X'].tolist() if not scaling else self.scale(data['X']).tolist()
        
        logging.debug("transforming y to list")
        self.y_train = data['y'].tolist()

        ## get feature_name
        self.feature_name = path.split('/')[-1].replace('.Xy','').split('.npz')[0]
        logging.debug("got feature_name %s" % (self.feature_name))

        ## if y_train looks like "happy" or "_happy"
        ## transform it to numeric expression such as 1 or 0
        if type(self.y_train[0]) not in (int , float):
            self.y_train = self.numerize(y=self.y_train)

    def load_test(self, path, scaling=False):

        data = np.load(path)

        _scaling_msg_ = "and scaling " if scaling else ""
        logging.debug("transforming %s X to list" % (_scaling_msg_))
        self.X_test = data['X'].tolist() if not scaling else self.scale(data['X']).tolist()
        
        logging.debug("transforming y to list")
        self.y_test = data['y'].tolist()
        

        self.feature_name = path.split('/')[-1].replace('.Xy','').split('.npz')[0]
        
        logging.debug("got feature_name %s" % (self.feature_name))

    def set_param(self, args="-t 0 -c 4 -b 1 -q"):
        self.param = self.svmutil.svm_parameter(args)
        return self.param

    def formulate(self):
        if not self.y_train or not self.X_train:
            logging.error("run load_train() to load training instances")
            return False
        self.prob = self.svmutil.svm_problem(self.y_train, self.X_train)

    def train(self, **kwargs):
        self.m = self.svmutil.svm_train(self.prob, self.param)

    def _build_classes_order(self, p_vals, p_labels, target):
        # e.g.,
        #  if p_val [0.53, 0.47] is labeled as p_label 0, then the order is [ 0, 1 ] --> [_happy, happy]
        #  if p_val [0.53, 0.47] is labeled as p_label 1, then the order is [ 1, 0 ] --> [happy, _happy]        
        #  if p_val [0.43, 0.57] is labeled as p_label 1, then the order is [ 0, 1 ] --> [_happy, happy]
        order = None
        for i in xrange(len(p_labels)):
            
            p_val, p_label = p_vals[i], p_labels[i]
            
            if p_val[0] == p_val[1]:
                continue
            else:
                maxidx = 0 if p_val[0] > p_val[1] else 1
                if maxidx == p_label:
                    order = [0, 1]
                    break
                else:
                    order = [1, 0]
                    break
        if not order :
            return False
        else:
            ## if the order is [ 0, 1 ]
            ## and the target is 'happy'
            ## then will return [ '_happy', 'happy' ]
            return [ "_"+target if _class == 0 else target for _class in order ]

    def predict(self, target, param=""):
        if not self.y_test or not self.X_test:
            logging.error("run load_test() to load training instances")
            return False

        ## relabel y_test
        logging.debug("relabeling y_test")
        ny_test = [1 if _y == target else 0 for _y in self.y_test]

        ## p_label, p_acc, p_val = svm_predict(y, x, m, '-b 1')
        logging.debug("predict X_test with options: %s" % (param))
        self.p_labels, self.p_acc, self.p_vals = self.svmutil.svm_predict(y=ny_test, x=self.X_test, m=self.m, options=param)
        
        # build classes order
        ## e.g., 
        #  if p_val [0.53, 0.47] is labeled as p_label 0, then the order is [ 0, 1 ] --> [_happy, happy]
        #  if p_val [0.53, 0.47] is labeled as p_label 1, then the order is [ 1, 0 ] --> [happy, _happy]
        self.classes_ = self._build_classes_order(self.p_vals, self.p_labels, target)

    def save_model(self, filename="auto", root="models", m=None):
        ## use abs "models" path instead
        path = os.path.join(root, ".".join([self.feature_name, "model"])) if filename == "auto" else os.path.join(root, filename)
        dirs = os.path.dirname(path)
        if dirs and not os.path.exists( dirs ): os.makedirs( dirs )  

        if not m: m = self.m
        self.svmutil.svm_save_model(path, m)

    def load_model(self, path):
        self.m = self.svmutil.svm_load_model(path)