# -*- coding: utf-8 -*-

def connect_mongo(mongo_addr='doraemon.iis.sinica.edu.tw', db_name='LJ40K'):
    import pymongo
    db = pymongo.Connection(mongo_addr)[db_name]
    return db

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

    NaN = -1 if 'NaN' not in kwargs else kwargs['NaN']
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

def RandomSample(arrays, dim=0.1):
    """
    Usage
    =====
        >> from feelit.utils import RandomSample
        >> (_X, _y) = RandomSample((X, y), 0.5) ## ratio version
        >> (_X, _y) = RandomSample((X, y), 100) ## set target dimension directly   

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
    
    ## check if n are all the same
    ns = [ len(array) if 'shape' not in dir(array) else array.shape[0] for array in arrays ]
    if len(set(ns)) != 1:
        ## n(s) differ in the input array(s)
        return False
    else:

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

        sampled = []
        for array in arrays:
            _array = np.delete(array, delete_indexes, axis=0)
            sampled.append( _array )

        return sampled




