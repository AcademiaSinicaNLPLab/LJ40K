import sys, argparse, os, csv
import numpy as np
sys.path.append( "../" )
from feelit import utils
from feelit.features import Learning
from feelit.features import DataPreprocessor
from sklearn.cross_validation import KFold
import operator
import logging
import pickle

emotions = utils.LJ40K

def get_arguments(argv):
    parser = argparse.ArgumentParser(description='perform SVM training for LJ40K')
    parser.add_argument('feature_list_file', metavar='feature_list_file', 
                        help='This program will fuse the features listed in this file and feed all of them to the classifier. The file format is in JSON. See "feautre_list_ex.json" for example')
    parser.add_argument('-k', '--kfold', metavar='NFOLD', type=int, default=10, 
                        help='k for kfold cross-validtion. If the value less than 2, we skip the cross-validation and choose the first parameter of -c and -g (DEFAULT: 10)')
    parser.add_argument('-o', '--output_file_name', metavar='OUTPUT_NAME', default='out.csv', 
                        help='path to the output file in csv format (DEFAULT: out.csv)')
    parser.add_argument('-e', '--emotion_ids', metavar='EMOTION_IDS', type=utils.parse_range, default=[0], 
                        help='a list that contains emotion ids ranged from 0-39 (DEFAULT: 0). This can be a range expression, e.g., 3-6,7,8,10-15')
    parser.add_argument('-c', metavar='C', type=utils.parse_list, default=[1.0], 
                        help='SVM parameter (DEFAULT: 1). This can be a list expression, e.g., 0.1,1,10,100')
    parser.add_argument('-g', '--gamma', metavar='GAMMA', type=utils.parse_list, default=None, 
                        help='RBF parameter (DEFAULT: 1/dimensions). This can be a list expression, e.g., 0.1,1,10,100')
    parser.add_argument('-t', '--temp_output_dir', metavar='TEMP_DIR', default=None, 
                        help='output intermediate data of each emotion in the specified directory (DEFAULT: not output)')
    parser.add_argument('-n', '--no_scaling', action='store_true', default=False,
                        help='do not perform feature scaling (DEFAULT: False)')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, 
                        help='show messages')
    parser.add_argument('-d', '--debug', action='store_true', default=False, 
                        help='show debug messages')
    args = parser.parse_args(argv)
    return args

def collect_results(all_results, emotion, results):
    all_results['emotion'].append(emotion)
    all_results['weighted_score'].append(results['weighted_score'])
    all_results['auc'].append(results['auc'])
    all_results['X_predict_prob'].append(results['X_predict_prob'])
    return all_results

if __name__ == '__main__':
    
    args = get_arguments(sys.argv[1:])
    features = utils.get_feature_list(args.feature_list_file)

    if args.debug:
        loglevel = logging.DEBUG
    elif args.verbose:
        loglevel = logging.INFO
    else:
        loglevel = logging.ERROR
    logging.basicConfig(format='[%(levelname)s][%(name)s] %(message)s', level=loglevel) 
    logger = logging.getLogger(__name__)

    # some pre-checking
    if args.temp_output_dir is not None and not os.path.isdir(args.temp_output_dir):
        raise Exception("temp folder %s doesn't exist." % (args.temp_output_dir))

    if os.path.exists(args.output_file_name):
        logger.warning("file %s will be overwrote." % (args.output_file_name))
    elif not utils.test_writable(args.output_file_name): 
        raise Exception("file %s is not writable." % (args.output_file_name))


    # main loop
    collect_best_param = {}   # TODO: remove
    all_results = {'emotion': ['Evals'], 'weighted_score': ['Accuracy Rate'], 'auc': ['AUC'], 'X_predict_prob': []}
    for emotion_id in args.emotion_ids:    
        
        emotion_name = emotions[emotion_id]
        paths = utils.get_paths_by_emotion(features, emotion_name)

        ## prepare data
        preprocessor = DataPreprocessor(loglevel=loglevel)
        preprocessor.loads([f['feature'] for f in features], paths)
        X_train, y_train, feature_name = preprocessor.fuse()

        ## set default gamma for SVM           
        if not args.gamma:
            args.gamma = [1.0/X_train.shape[1]]
                
        learner = Learning(loglevel=loglevel) 
        learner.set(X_train, y_train, feature_name)

        ## setup a kFolder
        if args.kfold > 1:
            kfolder = KFold(n=utils.getArrayN(X_train) , n_folds=args.kfold, shuffle=True)
        
            ## do kfold with Cs and gammas
            scores = {}
            for svmc in args.c:
                for rbf_gamma in args.gamma:
                    score = learner.kfold(kfolder, classifier='SVM', kernel='rbf', prob=False, C=svmc, scaling=(not args.no_scaling) ,gamma=rbf_gamma)
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
            fpath = os.path.join(args.temp_output_dir, "model_%s_%f_%F.pkl" % (emotion_name, best_C, best_gamma));
            learner.dump_model(fpath);
            if not args.no_scaling:
                fpath = os.path.join(args.temp_output_dir, "scaler_%s.pkl" % (emotion_name));
                learner.dump_scaler(fpath);

    if args.temp_output_dir:
        fpath = os.path.join(args.temp_output_dir, 'best_param.csv')
        utils.dump_dict_to_csv(fpath, collect_best_param)
        fpath = os.path.join(args.temp_output_dir, 'X_predict_prob.csv')    
        utils.dump_list_to_csv(fpath, all_results['X_predict_prob'])
        

    utils.dump_list_to_csv(args.output_file_name, [all_results['emotion'], all_results['weighted_score'], all_results['auc']])   
