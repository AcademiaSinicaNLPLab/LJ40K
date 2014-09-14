
import logging, os
import numpy as np
from collections import defaultdict

class LateFusion(object):
    """
    from feelit.Evaluations import LateFusion

    lf = LateFusion(verbose=True)

    lf.loads(root="results/TFIDF+keyword.classifier=LIBSVM_classtype=binary_kernel=rbf_prob=True.results")

    lf.loads(root="results/rgba_gist+rgba_phog.classifier=LIBSVM_classtype=binary_kernel=linear_prob=True.results")

    lf.fuse()
    """
    def __init__(self, **kwargs):
        loglevel = logging.DEBUG if 'verbose' in kwargs and kwargs['verbose'] == True else logging.INFO
        logging.basicConfig(format='[%(levelname)s] %(message)s', level=loglevel)

        ## put False label at the first column, True at the second.
        ## e.g.,
        ##  classes: [ 'happy', '_happy' ]
        ##  results: [ [ 0.4, 0.6 ], ... ] 
        ##
        ##       --> 
        ##  classes: [ 'happy', '_happy' ],
        ##  results: [ [ 0.6, 0.4 ], ... ]        
        self._order = [0, 1]
        logging.debug("order of classes is [%d, %d]" % (self._order[0], self._order[1]))
        
        self.results_lst_dict = defaultdict(list)
        self.answers = None

    def swap(self, arr):
        ## swap column 0 and column 1
        return arr[:,[1, 0]]

    def loads(self, root):
        ## root
        # "TFIDF+keyword.classifier=LIBSVM_classtype=binary_kernel=linear_prob=True.results"

        ## npz_files
        # rgba_gist+rgba_phog.tired.results.npz
        npz_files = filter(lambda x:x.endswith(".npz"), os.listdir(root) )

        logging.info("loading npz files in %s" % root)
        for npz_file in npz_files:
            path = os.path.join(root, npz_file)
            self.load(path)

    def load(self, path):
        """
        Load a result (.npz) file generated in testing stage

        Parameters
        ==========
        path: str
            path to the .npz file

        Data
        ====

        ## data.files will return
        # ['classes', 'results', 'answers']

        # data['classes']:
        # array(['_tired', 'tired'],
        #       dtype='|S6')

        # data['results'] looks like:
        # array([[ 0.34265826,  0.65734174],
        #        [ 0.65432564,  0.34567436],
        #        [ 0.51869872,  0.48130128],
        #        ...,
        #        [ 0.73026024,  0.26973976],
        #        [ 0.71746164,  0.28253836],
        #        [ 0.45684861,  0.54315139]])

        # data['answers']:
        # array(['accomplished', 'accomplished', 'accomplished', ..., 'tired',
        #        'tired', 'tired'],
        #       dtype='|S13') 

        Returns
        =======
        no return value. the loaded data are saved in `self.results_lst_dict`

        """
        data = np.load(path)

        if type(self.answers) != type(None):
            self.answers = data['answers'] ## save answers of testing data (execute once)

        ## check if the order is correct
        false_index_order = self._order.index(0)
        false_index_label = 0 if data['classes'][0].startswith('_') else 1

        if false_index_order != false_index_label: ## need to swap
            logging.debug("the classes are in the wrong order (%s, %s) --> (%s, %s)" % (data['classes'][0], data['classes'][1], data['classes'][1], data['classes'][0]))
            results = self.swap(data['results'])
        else:
            results = data['results']

        ## no matter the first classes is "happy" or "_happy"
        ## will got the label "happy"
        label = data['classes'][0].replace('_','')

        ## push to a list by label
        self.results_lst_dict[label].append( results )

    def fuse(self, weights="average"):
        """
        Fuse all loaded results

        Parameters
        ==========

        weights: "average" or list or tuple
            the sequence of weights for each candidate

        """
        ## check size

        if len(set([len(lst) for lst in self.results_lst_dict.values()])) != 1:
            logging.error("check the number of labels in each candidate")
            return False

        ## deal with weight
        type_nums = len(self.results_lst_dict.values()[0])
        if weights == "average":
            weights = [1/float(type_nums)]*type_nums

        ### check weight alignment
        if len(weights) != type_nums:
            logging.error("make sure the # of weights is the same as # of candidates")
            return False
        
        ### normalize weight
        W = float(sum(weights))
        if W == 0.0:
            logging.error("plz assign a non-zero weight sequence")
            return False
        weights = [w/W for w in weights]

        ## start to fuse
        self.fused = {}
        for label, candidate_lst in self.results_lst_dict.iteritems():
            self.fused[label] = reduce(lambda x,y: x+y, [arr*w for arr, w in zip(candidate_lst, weights)] )

        return self.fused


class Evaluation(object):
    """
    from feelit.Evaluations import Evaluation
    ev = Evaluation(verbose=True)
    ev.loads(root="results/text_TFIDF.classifier=SGD_classtype=binary_kernel=linear_prob=True.results")
    ev.eval_all()
    ev.save(root="evals")
    """
    def __init__(self, **kwargs):
        loglevel = logging.DEBUG if 'verbose' in kwargs and kwargs['verbose'] == True else logging.INFO
        logging.basicConfig(format='[%(levelname)s] %(message)s', level=loglevel)

        self.PN = {}
        self.feature_name = None

    def loads(self, root="results/text_TFIDF.classifier=SGD_classtype=binary_kernel=linear_prob=True.results"):
        
        ## extract "text_TFIDF.classifier=SGD_classtype=binary_kernel=linear_prob=True"
        self.settings = root.split("/")[-1].split(".results")[0]

        npz_files = filter(lambda x: x.endswith(".npz"), os.listdir(root))

        self.PNs = {}
        self.ratios = {}

        for npz_file in npz_files:

            path = os.path.join(root, npz_file)
            data = np.load(path)

            cls = data['classes']

            positive_idx = 1 if cls[0].startswith("_") else 0
            negative_idx = 1-positive_idx

            target = cls[positive_idx]

            TP, TN, FP, FN = 0, 0, 0, 0
            really_is_positive, really_is_negative = 0, 0
            for res, ans in zip( data['results'], data['answers']):

                classified_as = "POS" if res[positive_idx] > res[negative_idx] else "NEG"

                really_is = "POS" if ans == target else "NEG"

                really_is_positive += 1 if really_is == "POS" else 0
                really_is_negative += 1 if really_is == "NEG" else 0

                # print res, ans, ' --> classified_as', classified_as, '; really_is', really_is

                TP += 1 if classified_as == "POS" and really_is == "POS" else 0
                TN += 1 if classified_as == "NEG" and really_is == "NEG" else 0
                FP += 1 if classified_as == "POS" and really_is == "NEG" else 0
                FN += 1 if classified_as == "NEG" and really_is == "POS" else 0

            self.ratios[target] = really_is_negative/float(really_is_positive)
            self.PNs[target] = { "TP": TP, "TN": TN, "FP": FP, "FN": FN }
            
    def eval_all(self):
        self.scores = { label: self.accuracy(self.PNs[label], self.ratios[label]) for label in self.PNs }
        self.avg = sum(self.scores.values())/float(len(self.scores))

    def save(self, root="performances"):
        
        if not os.path.exists(root): os.makedirs(root)
        out_path = os.path.join(root, self.settings+'.eval.npz')

        logging.debug("saving evalution scores")
        np.savez_compressed(out_path, scores=self.scores, avg=self.avg, PNs=self.PNs, ratios=self.ratios, settings=self.settings)

    def accuracy(self, PN, ratio):
        TP = PN['TP']
        TN = PN['TN']/float(ratio)
        FP = PN['FP']/float(ratio)
        FN = PN['FN']
        return round((TP+TN)/float(TP+TN+FN+FP), 4)
            
