
import logging, os
import numpy as np

class LateFusion(object):
    """docstring for LateFusion"""
    def __init__(self):
        pass
    def load():
        pass
        


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
            
