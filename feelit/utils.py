# -*- coding: utf-8 -*-

from math import exp
import json
import os

LJ40K = ['accomplished', 'aggravated', 'amused', 'annoyed', 'anxious', 'awake', 'blah', 'blank', 'bored', 'bouncy', 'busy', 'calm', 'cheerful', 'chipper', 'cold', 'confused', 'contemplative', 'content', 'crappy', 'crazy', 'creative', 'crushed', 'depressed', 'drained', 'ecstatic', 'excited', 'exhausted', 'frustrated', 'good', 'happy', 'hopeful', 'hungry', 'lonely', 'loved', 'okay', 'pissed off', 'sad', 'sick', 'sleepy', 'tired']

def connect_mongo(mongo_addr='doraemon.iis.sinica.edu.tw', db_name='LJ40K'):
    import pymongo
    db = pymongo.Connection(mongo_addr)[db_name]
    return db

def all_to_float(two_d_array, mapping={None: 0.0, "NaN": 0.0, "null": 0.0}, extra={}):
    """
    Parameters
    ==========
    two_d_array: numpy.array() with shape(_, _) or <list of list>

    mapping: dict
        default mapping table

    extra: dict
        extra mapping options

    Returns
    =======
    nothing, this is an in-place function

    """
    mapping.update(extra)
    for rid in xrange(len(two_d_array)):
        for cid in xrange(len(two_d_array[rid])):
            if type(two_d_array[rid][cid]) not in (float, int):
                if two_d_array[rid][cid] in mapping:
                    two_d_array[rid][cid] = mapping[ two_d_array[rid][cid] ]
                else:
                    print "unknown mapping of type: ", two_d_array[rid][cid], '(type:', type(two_d_array[rid][cid]), ')'
                    raise TypeError
            else:
                pass



def toNumber(string, **kwargs):
    """
    Convert a string to number
    string:
        <str>, input string
    kwargs:
        NaN:
            <int>/<float>, convert "NaN" to certain value
        toFloat:
            "auto"/True/False, set auto to detect "." in the input string automatically
    """

    NaN = 0 if 'NaN' not in kwargs else kwargs['NaN']
    toFloat = "auto" if 'toFloat' not in kwargs else kwargs['toFloat']

    string = string.strip()
    if "nan" in string.lower():
        return NaN
    elif toFloat == True or toFloat == "auto" and "." in string:
        return float(string)
    elif toFloat == False:
        return int(string)

def find_missing(root):
    """
    find documents without features in emotion-image output
    Parameter:
        root: the root path of the nested folders
    Returns:
        a dict containing missing documents
    """
    import os
    from collections import defaultdict
    Missing = {}
    for i, folder in enumerate(filter(lambda x:x[0]!='.', os.listdir(root))):
        files = set(map(lambda x: int(x.split('/')[-1].split('.')[0]), os.listdir( os.path.join(root, folder) )))

        ## global     local
        ## -----------------
        ## 0~999      0~999
        ## 1000~1999  0~999
        ## ...        ...
        label = folder
        Missing[label] = defaultdict(list)
        for (_local_id, _global_id) in enumerate(range(i*1000,(i+1)*1000)):
             if _global_id not in files:
                Missing[label]['global'].append(_global_id)
                Missing[label]['local'].append(_local_id)
    return Missing

def load_csv(path, **kwargs):

    LINE = "\n" if 'LINE' not in kwargs else kwargs['LINE']
    ITEM = "," if 'ITEM' not in kwargs else kwargs['ITEM']
    number = True if 'number' not in kwargs else kwargs['number']
 
    doc = open(path).read()
    lines_raw = doc.strip().split(LINE)

    if number:
        lines = [ map(lambda x: toNumber(x, **kwargs), line.split(ITEM) ) for line in lines_raw]
    else:
        lines = [line.split(ITEM) for line in lines_raw]
    return lines

def amend(missing, input_path, output_path, **kwargs):
    import os
    # e.g.,
    # input_path = '/Users/Maxis/projects/emotion-detection-modules/dev/image/emotion_imgs_threshold_1x1_rbga_out/out_f1'
    # output_path = '/Users/Maxis/projects/emotion-detection-modules/dev/image/emotion_imgs_threshold_1x1_rbga_out_amend/out_f1'
    """
    input_path: <str: folder> 
    output_path: <str: folder>
    """
    for fn in os.listdir(input_path):

        fn_path = os.path.join(input_path, fn)
        out_path = os.path.join(output_path, fn)
        ## get label
        label, feature_name = fn_path.split('/')[-1].split('.')[0].split('_')

        lines = load_csv(fn_path, **kwargs)

        new_lines = []
        for line in lines:
            for idx in missing[label]['local']:
                line.insert(idx, 'NaN')
            new_lines.append( line )

        content = LINE.join([ ITEM.join(nl) for nl in new_lines])

        with open(out_path, 'w') as fw:
            fw.write(content)

def getArrayN(array):
    """
    Get number of rows in the input array
    """
    n = False
    try:
        ## dense matrix
        n = len(array)
    except TypeError:
        ## sparse matrix
        n = array.shape[0]
    return n

def getShape(arr):
    return arr.shape if arr.shape else arr.any().shape


def isSparse(array):
    """
    Detect if input is a sparse array or not
    """
    try:
        ## dense matrix
        len(array)
    except TypeError:
        ## sparse matrix
        return True
    else:
        return False

def toDense(a):
    if not a.shape:     
        # this is a trick invented by Maxis.
        # when a csr_matrix saved by savez_compressed is loaded by np.load,
        # we need to use all() to convert the matrix back to csr_matrix type
        densed_a = a.all().toarray()
    elif isSparse(a):
        densed_a = a.toarray()
    else:
        densed_a = a
    return densed_a

def GenerateDeleteIndexes(n, dim, path=None):
    """
    Usage
    =====
        >> from feelit.utils import GenerateDeleteIndexes
        ## write to file
        >> GenerateDeleteIndexes(n=32000, dim=3200, path="idxs.pkl")
        ## directly return
        >> GenerateDeleteIndexes(n=32000, dim=3200)
    """
    import random
    # get all indexes
    all_indexes = range(n) 
    # shuffle them and choose [0:dim]
    random.shuffle(all_indexes)
    choosen_indexes = set(all_indexes[:dim])
    delete_indexes = [idx for idx in all_indexes if idx not in choosen_indexes]

    if path:
        import pickle
        pickle.dump( delete_indexes, open(path, "w"), protocol=pickle.HIGHEST_PROTOCOL )
    else:
        return delete_indexes

def strShape(X):
    return 'x'.join(map(lambda x:str(x), X.shape))

def squared_Euclidean_distance(p, q):
    return sum([ (_p-_q)*(_p-_q) for _p, _q in zip(p, q)])

def rbf_kernel_function(v1, v2, gamma="default"):
    if gamma == "default":
        num_features = len(v1)
        gamma = 1.0/num_features

    sed = squared_Euclidean_distance(v1, v2)
    k_v1_v2 = exp(-1.0*gamma*sed)
    return k_v1_v2

def devide(X, part, shuffle=False):
    """
    Devide X (array, matrix or list) into subsets according to the percent

    Usage
    =====
    >> from feelit.utils import devide
    >> devide(X, 0.5)
    >> devide(X, 100)
    >> devide(X, 100, shuffle=True)

    Parameters
    ==========
    X: array, matrix or list

    part: int or float

    random: boolean

    Returns
    =======
    devided X: tuple

    """
    n = len(X)
    endpoint = -1
    # deal with part
    if type(part) == float and part < 1.0 and part > 0:
        endpoint = n*part
    elif type(part) == float and ( part >= 1.0 or part <= 0):
        raise Exception('the value of `part` must lie in the range from 0 ot 1')        
    elif type(part) == int and part < n:
        endpoint = part
    elif type(part) == int and part >= n:
        raise Exception('the value of `part` must less than total samples')
    else:
        raise Exception('check the value and type of `part`, it must be int or float')

    if shuffle:
        import random
        random.shuffle(X)

    return (X[:endpoint], X[endpoint:])
      
def random_idx(length, topk):
    import random
    # generate index for random
    all_indexes = range(length)
    random.shuffle(all_indexes)
    return (all_indexes[:topk], all_indexes[topk:])

def RandomSample(arrays, dim=0.1, delete_index=None):
    """
    Usage
    =====
        >> from feelit.utils import RandomSample
        >> (_X, _y) = RandomSample((X, y), 0.5) ## ratio version
        >> (_X, _y) = RandomSample((X, y), 100) ## set target dimension directly
        >> (_X, _y) = RandomSample((X, y), delete_index="data/idxs.pkl") ## specify a certain list of indexes to be deleted

    Parameters
    ==========
        S: list/tuple of numpy.ndarray(s)
            a numpy array
        dim: float or int, default 0.1
            target dimension
            use float to set ratio
            e.g.,
                S is a <1024 x 10> matrix
                dim = 0.1
                target dimension will be 1024*0.1 = 102.4 --> 102
                floor function is used by default
    Returns
    =======
    sampled_S: list of numpy.ndarray
        a random down-sampled array
    """
    # random sampling
    import random
    import numpy as np
    import os
    import pickle

    ## check if n are all the same
    ns = [ len(array) if 'shape' not in dir(array) else array.shape[0] for array in arrays ]
    if len(set(ns)) != 1:
        ## n(s) differ in the input array(s)
        return False
    else:

        delete_indexes = []

        if delete_index:
            if type(delete_index) == str:
                if os.path.exists(delete_index):
                    delete_indexes = pickle.load(open(delete_index))
            elif type(delete_index) == list:
                delete_indexes = delete_index
            elif type(delete_index) == np.ndarray:
                delete_indexes = list(delete_index)
            else:
                pass

        if not delete_indexes:
            # get number of samples
            n = ns[0]

            # set dim
            if type(dim) == int and dim < n:
                pass
            elif type(dim) == int and dim >= n:
                return False
            elif type(dim) == float:
                dim = int(dim*n)
                if dim == 0:
                    return False

            # get all indexes
            all_indexes = range(n) 

            # shuffle them and choose [0:dim]
            random.shuffle(all_indexes)
            choosen_indexes = set(all_indexes[:dim])
            delete_indexes = [idx for idx in all_indexes if idx not in choosen_indexes]

            ## save to path of `index_file`
            if index_file:
                dest_dir = os.path.dirname(index_file)
                if not os.path.isdir(dest_dir):
                    os.makedirs(dest_dir)
                pickle.dump(delete_indexes, open(index_file, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

        sampled = []
        for array in arrays:
            _array = np.delete(array, delete_indexes, axis=0)
            sampled.append( _array )

        return sampled

def dump_dict_to_csv(file_name, data):
    import csv
    w = csv.writer(open(file_name, 'w'))
    for key, val in data.items():
        w.writerow([key, val])

def dump_list_to_csv(file_name, data):
    import csv
    w = csv.writer(open(file_name, 'w'))
    for row in data:
        w.writerow(row)

############################################## arguments parsing
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

def test_writable(file_path):
    writable = True
    try:
        filehandle = open(file_path, 'w')
    except IOError:
        writable = False
        
    filehandle.close()
    return writable

