# -*- coding: utf-8 -*-
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import numpy as np
import logging

class Metrics(object):
    """docstring for Metrics"""
    def __init__(self):
        pass

    def accuracy(self, res, ratio=1):
        TP = res['TP']
        TN = res['TN']/float(ratio)
        FP = res['FP']/float(ratio)
        FN = res['FN']
        return round((TP+TN)/float(TP+TN+FN+FP), 4)

    def precision(self, res, ratio=1):
        TP = res['TP']
        TN = res['TN']/float(ratio)
        FP = res['FP']/float(ratio)
        FN = res['FN']
        return round((TP)/float(TP+FP), 4)

    def recall(self, res, ratio=1):
        TP = res['TP']
        TN = res['TN']/float(ratio)
        FP = res['FP']/float(ratio)
        FN = res['FN']
        return round((TP)/float(TP+FN), 4)        
        

class Evaluation(object):
    """
    Evaluation
    
    Evaluate prediction results of input samples of classifiers
    output the evaluation scores
    """

    performaces = defaultdict(dict)
    aggregated = dict()

    def __init__(self, **kwargs):

        loglevel = logging.DEBUG if 'verbose' in kwargs and kwargs['verbose'] == True else logging.INFO
        logging.basicConfig(format='[%(levelname)s] %(message)s', level=loglevel)

        self.feature_name = None if 'feature_name' not in kwargs else kwargs['feature_name']
        self.ans_pre = [] if 'ans_pre' not in kwargs else kwargs['ans_pre']

    def load(self, fn, delimiter=","):
        self.fn = fn
        self.ans_pre = zip(*map(lambda x:x.split(delimiter), open(self.fn).read().strip().split('\n')))

        logging.debug('%s loaded' % (fn))

    def _getTargets(self):
        targets = set()
        for ans, pre in self.ans_pre:
            targets.add(ans)
        return sorted(list(targets))

    def evals(self):
        if not self.ans_pre:
            raise TypeError('the list of answer-predict pairs is not given')
            return False

        targets = self._getTargets()

        POSITIVE, NEGATIVE = True, False

        self.results = {}

        logging.debug('evaluate %d targets' % (len(targets)))

        for target in targets:

            really_is_positive, really_is_negative = 0, 0

            # print '>> current target:', target

            instances = Counter()

            for ans, pre in self.ans_pre:

                really_is = POSITIVE if target == ans else NEGATIVE
                classified_as = POSITIVE if target == pre else NEGATIVE


                ## stat really_is_Positive: really_is_Negative = 200: 7900
                really_is_positive += 1 if really_is == POSITIVE else 0
                really_is_negative += 1 if really_is == NEGATIVE else 0

                TP = classified_as == POSITIVE and really_is == POSITIVE
                TN = classified_as == NEGATIVE and really_is == NEGATIVE
                FP = classified_as == POSITIVE and really_is == NEGATIVE
                FN = classified_as == NEGATIVE and really_is == POSITIVE

                instances['TP'] += 1 if TP else 0
                instances['TN'] += 1 if TN else 0
                instances['FP'] += 1 if FP else 0
                instances['FN'] += 1 if FN else 0

            r = really_is_negative/float(really_is_positive)

            self.results[target] = {
                'instances': instances,
                'ratio': r
            }

        

    def measure(self):

        #########################################################
        # performaces:
        # {
        #     'results/rgb-f1-r1/rgb-f1-r1.fold-1.result':
        #     {
        #         'accomplished': 0.4986,
        #         'aggravated': 0.5,
        #         'amused': 0.5049,
        #         'annoyed': 0.5043, ...            
        #     },
        #     ...

        #     'results/rgb-f1-r1/rgb-f1-r1.fold-10.result': 
        #     {   
        #         'accomplished': 0.4986,
        #         'aggravated': 0.5,
        #         'amused': 0.5049,
        #         'annoyed': 0.5043, ...
        #     }
        # }
        #########################################################
        
        me = Metrics()
        logging.debug('apply metrics')
        for target in self.results:

            ## choose one measure
            accu = me.accuracy(res=self.results[target]['instances'], ratio=self.results[target]['ratio'])
            self.performaces[self.fn][target] = accu

    def add(self, fn):
        """
        a wrapper of load, evals and measure
        """

        self.load(fn)
        self.evals()
        self.measure()

        logging.info('%s successfully added' % (fn))

        
    def aggregate(self):
        logging.debug('average each fold')

        logging.debug('grouping by emotions')        
        # group_by_emotion:
        # {
        #     'crazy': [0.5015, 0.4994, 0.5, 0.5013, 0.4999, 0.5, 0.5, 0.5, 0.4831, 0.5],
        #     'tired': [0.4987, 0.5, 0.5263, 0.5044, 0.5, 0.5098, 0.5136, 0.5, 0.5, 0.499],
        #     ...
        # }
        group_by_emotion = defaultdict(list)
        for fn in self.performaces:
            for target in self.performaces[fn]:
                group_by_emotion[target].append( self.performaces[fn][target] )

        ## aggregate each group
        aggregated = { x : sum(group_by_emotion[x])/float(len(group_by_emotion[x])) for x in group_by_emotion }


        targets = []
        try:
            targets = sorted(aggregated.keys(), key=lambda x:int(x))
        except ValueError:
            targets = sorted(aggregated.keys())

        # target_map = {
        #  'accomplished': 0,
        #  'aggravated': 1,
        #  'amused': 2,
        #  'annoyed': 3,
        #  'anxious': 4,
        #  'awake': 5,
        #  ...
        # }
        self.target_map = {t : i for i,t in enumerate(targets)}

        self.x, self.y = [], []
        for target in targets:
            self.x.append(self.target_map[target])
            self.y.append(aggregated[target])

        ## calculate total average
        self.total_average = sum(self.y)/float(len(self.y))


    def setFeatureName(self, feature_name):
        self.feature_name = feature_name

    def plot(self, feature_name=None, labels={}, **kwargs):
        """
        kwargs:
           xlabel: Emotion
           ylabel: Accuracy
           color: random, pyplot.color
        """
        if feature_name:
            self.feature_name = feature_name

        if not self.feature_name:
            raise Exception('set the feature_name by passing the parameter <feature_name> or using setFeatureName()')

        _xlabel = 'Emotion' if 'xlabel' not in kwargs else kwargs['xlabel']
        _ylabel = 'Accuracy' if 'ylabel' not in kwargs else kwargs['ylabel']

        ## deal with coloring
        if 'color' in kwargs:
            _color = np.random.rand(3,1) if kwargs['color'] == 'random' else kwargs['color']
            plt.plot(self.x, self.y, c=_color)
        else:
            plt.plot(self.x, self.y)

        plt.title("%s over %d emotions (%s) %.2f%%" % (_ylabel, len(self.target_map), self.feature_name, self.total_average*100))

        plt.xlabel(_xlabel)
        plt.ylabel(_ylabel)

        if not labels:
            self.labels = { self.target_map[t] : t for t in self.target_map}
        else:
            self.labels = label

        _labels = [ self.labels[_x] for _x in self.x ]

        plt.xticks(range(len(_labels)), _labels, rotation=90)

        plt.tight_layout()

        plt.grid()
        plt.xlim([min(self.x), max(self.x)])


    def save_plot(self, out_fn="$default", ext="png"):
        if out_fn == "$default":
            out_fn = self.feature_name
        if not out_fn:
            raise Exception("Please specified the output filename by setting <out_fn>")
        plt.savefig('.'.join([out_fn, ext]))
        plt.close()

    def save_txt(self, out_fn="$default", ext="txt"):
        if out_fn == "$default":
            out_fn = self.feature_name
        if not out_fn:
            raise Exception("Please specified the output filename by setting <out_fn>")

        # _labels = [ labels[_x] for _x in self.x ]
        with open('.'.join([out_fn, ext]), 'w') as fw:

            for x, y in zip(self.x, self.y):
                fw.write( "%s\t%f\n" % ( self.labels[x], y) )
                
if __name__ == '__main__':

    # from Evaluation import Evaluation

    import pymongo
    labels = { i: e for i, e in enumerate(sorted(map(lambda x:x['emotion'], pymongo.Connection('doraemon.iis.sinica.edu.tw')['LJ40K']['emotions'].find({'label': 'LJ40K'}))))}

    ev = Evaluation(feature_name="image-RGBA-f2")

    for i in range(10): ev.add('results/rgba-f2/rgba-f2.fold-%d.result' % (i+1))
    # for i in range(10): ev.add('features.DepPairs.fold-%d.result' % (i+1))

    ev.aggregate()

    ev.plot(xlabel="Emotion", ylabel="Accuracy")
    
    ev.save_plot()
    ev.save_txt()
    # ev.save(out_fn="TFIDF", ext="png")
    # ev.save(out_fn="DepPairs", ext="png")
    

