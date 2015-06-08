import sys, argparse, os
sys.path.append( "../" )
from feelit import utils
from feelit.features import Learning
from feelit.features import DataPreprocessor
import logging

emotions = utils.LJ40K

def get_arguments(argv):
    parser = argparse.ArgumentParser(description='load a trained model and predict the results')
    parser.add_argument('model_file_name', metavar='MODEL_FILE', 
                        help='input model file')
    parser.add_argument('emotion_id', metavar='EMOTION_ID', type=int,  
                        help='0-39, go check utils.LJ40K')
    parser.add_argument('feature_list_file', metavar='feature_list_file', 
                        help='This program will fuse the features listed in this file. This program will load the testing file only.')
    parser.add_argument('-s', '--scaler_file', metavar='SCALER_FILE', default=None, 
                        help='scaler file for scaling')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, 
                        help='show messages')
    parser.add_argument('-d', '--debug', action='store_true', default=False, 
                        help='show debug messages')
    args = parser.parse_args(argv)
    return args

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

    # pre-checking
    if not os.path.exists(args.model_file_name):
        raise Exception("model file %s doesn't exist." % (args.model_file_name))
    if not os.path.exists(args.feature_list_file):
        raise Exception("feature file %s doesn't exist." % (args.feature_list_file))    


    #load
    learner = Learning(loglevel=loglevel) 
    if args.scaler_file:
        learner.load_scaler(args.scaler_file)
    learner.load_model(args.model_file_name)

    # prepare test data
    paths = [f['test_file'] for f in features]
    preprocessor = DataPreprocessor(loglevel=loglevel)
    preprocessor.loads([f['feature'] for f in features], paths)
    X_test, y_test, feature_name = preprocessor.fuse()    
    emotion_name = emotions[args.emotion_id]
    yb_test = preprocessor.get_binary_y_by_emotion(y_test, emotion_name)

    # predict
    results = learner.predict(X_test, yb_test, weighted_score=True, X_predict_prob=True, auc=True)

    # ToDo: write a result file
