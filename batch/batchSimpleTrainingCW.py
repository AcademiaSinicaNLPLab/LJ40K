import sys, getopt, json, argparse, os, csv
import numpy as np
sys.path.append( "../" )
from feelit import utils
from feelit.features import Learning
from feelit.features import DataPreprocessor
from sklearn.cross_validation import KFold
import operator
import logging

emotions = utils.LJ40K

def parse_range(astr):
    result = set()
    for part in astr.split(','):
        x = part.split('-')
        result.update(range(int(x[0]), int(x[-1]) + 1))
    return sorted(result)

def parse_list(astr):
    result = set()
    for part in astr.split(','):
        result.add(float(part))
    return sorted(result)

def get_arguments(argv):
    parser = argparse.ArgumentParser(description='perform SVM training for LJ40K')
    parser.add_argument('feature_list_file', metavar='feature_list_file', 
                        help='This program will fuse the features listed in this file and feed all of them to the classifier. The file format is in JSON. See "feautre_list_ex.json" for example')
    parser.add_argument('-k', '--kfold', metavar='NFOLD', type=int, default=10, 
                        help='k for kfold cross-validtion. If the value less than 2, we skip the cross-validation and choose the first parameter of -c and -g (DEFAULT: 10)')
    parser.add_argument('-o', '--output_file_name', metavar='OUTPUT_NAME', default='out.csv', 
                        help='path to the output file in csv format (DEFAULT: out.csv)')
    parser.add_argument('-e', '--emotion_ids', metavar='EMOTION_IDS', type=parse_range, default=[0], 
                        help='a list that contains emotion ids ranged from 0-39 (DEFAULT: 0). This can be a range expression, e.g., 3-6,7,8,10-15')
    parser.add_argument('-c', metavar='C', type=parse_list, default=[1.0], 
                        help='SVM parameter (DEFAULT: 1). This can be a list expression, e.g., 0.1,1,10,100')
    parser.add_argument('-g', '--gamma', metavar='GAMMA', type=parse_list, default=None, 
                        help='RBF parameter (DEFAULT: 1/dimensions). This can be a list expression, e.g., 0.1,1,10,100')
    parser.add_argument('-t', '--temp_output_dir', metavar='TEMP_DIR', 
                        help='output intermediate data of each emotion in the specified directory (DEFAULT: not output)')
    parser.add_argument('-n', '--no_scaling', action='store_true', default=False,
                        help='do not perform feature scaling (DEFAULT: False)')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, 
                        help='show messages')
    parser.add_argument('-d', '--debug', action='store_true', default=False, 
                        help='show debug messages')
    args = parser.parse_args(argv)
    return args
    
def get_feature_list(feature_list_file):
    fp = open(feature_list_file, 'r')
    feature_list = json.load(fp)
    fp.close()
    return feature_list

def get_file_name_by_emtion(train_dir, emotion, **kwargs):
    '''
    serach the train_dir and get the file name with the specified emotion and extension
    '''
    ext = '.npz' if 'ext' not in kwargs else kwargs['ext']
    files = [fname for fname in os.listdir(train_dir) if os.path.isfile(os.path.join(train_dir, fname))]

    # target file is the file that contains the emotion string and has the desginated extension
    for fname in files:
        target = None
        if fname.endswith(ext) and fname.find(emotion) != -1:
            target = fname
            break
    return target

def get_paths_by_emotion(features, emotion_name):
    paths = []
    for feature in features:         
        fname = get_file_name_by_emtion(feature['train_dir'], emotion_name, exp='.npz')
        if fname is not None:
            paths.append(os.path.join(feature['train_dir'], fname))
    return paths

def collect_results(all_results, emotion, results):
    all_results['emotion'].append(emotion)
    all_results['weighted_score'].append(results['weighted_score'])
    all_results['auc'].append(results['auc'])
    all_results['X_predict_prob'].append(results['X_predict_prob'])
    return all_results

if __name__ == '__main__':
    
    args = get_arguments(sys.argv[1:])
    features = get_feature_list(args.feature_list_file)

    if args.debug:
        loglevel = logging.DEBUG
    elif args.verbose:
        loglevel = logging.INFO
    else:
        loglevel = logging.ERROR
    logging.basicConfig(format='[%(levelname)s] %(message)s', level=loglevel) 

    # main loop
    collect_best_param = {}   # TODO: remove
    all_results = {'emotion': ['Evals'], 'weighted_score': ['Accuracy Rate'], 'auc': ['AUC'], 'X_predict_prob': []}

    ## prepare data
    paths = [f['train_file'] for f in features]

    preprocessor = DataPreprocessor(logger=logging)
    preprocessor.loads([f['feature'] for f in features], paths)
    X_train, y_train, feature_name = preprocessor.fuse()

    for emotion_id in args.emotion_ids:    
        
        emotion_name = emotions[emotion_id]
        
        yb_train = preprocessor.get_binary_y_by_emotion(y_train, emotion_name)

        ## set default gamma for SVM           
        if not args.gamma:
            args.gamma = [1.0/X_train.shape[1]]
                
        learner = Learning(logger=logging) 
        learner.set(X_train, yb_train, feature_name)

        ## setup a kFolder
        if args.kfold > 1:
            kfolder = KFold(n=utils.getArrayN(X_train) , n_folds=args.kfold, shuffle=True)
        
            ## do kfold with Cs and gammas
            scores = {}
            for svmc in args.c:
                for rbf_gamma in args.gamma:
                    score = learner.kfold(kfolder, classifier='SVM', kernel='rbf', prob=False, C=svmc, scaling=(not args.no_scaling), gamma=rbf_gamma)
                    scores.update({(svmc, rbf_gamma): score})

            if args.temp_output_dir:
                fpath = os.path.join(args.temp_output_dir, 'scores_%s.csv' % emotion_name)
                utils.dump_dict_to_csv(fpath, scores)

            ## get best parameters
            best_C, best_gamma = max(scores.iteritems(), key=operator.itemgetter(1))[0]

            ## collect misc
            collect_best_param.update({emotion_name: (best_C, best_gamma)}) 

        else:   # we choose first parameters if we do not perfrom cross-validation
            best_C = args.c[0] if args.c else 1.0
            best_gamma = args.gamma[0] if args.gamma else (1.0/X_train.shape[1])


        ## ---------------------------------------------------------------------------
        ## train all data
        learner.train(classifier='SVM', kernel='rbf', prob=True, C=best_C, gamma=best_gamma, scaling=(not args.no_scaling), random_state=np.random.RandomState(0))

        ## prepare testing data
        paths = [f['test_file'] for f in features]
        preprocessor.clear()
        preprocessor.loads([f['feature'] for f in features], paths)
        X_test, y_test, feature_name = preprocessor.fuse()
        
        yb_test = preprocessor.get_binary_y_by_emotion(y_test, emotion_name)
        results = learner.predict(X_test, yb_test, weighted_score=True, X_predict_prob=True, auc=True)

        ## collect results
        all_results = collect_results(all_results, emotion_name, results)

    if args.temp_output_dir:
        fpath = os.path.join(args.temp_output_dir, 'best_param.csv')
        utils.dump_dict_to_csv(fpath, collect_best_param)
        fpath = os.path.join(args.temp_output_dir, 'X_predict_prob.csv')    
        utils.dump_list_to_csv(fpath, all_results['X_predict_prob'])

    utils.dump_list_to_csv(args.output_file_name, [all_results['emotion'], all_results['weighted_score'], all_results['auc']])   
