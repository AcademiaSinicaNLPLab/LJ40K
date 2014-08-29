# -*- coding: utf-8 -*-

def connect_mongo(mongo_addr='doraemon.iis.sinica.edu.tw', db_name='LJ40K'):
    import pymongo
    db = pymongo.Connection(mongo_addr)[db_name]
    return db

def toNumber(string, NaN=-1, toFloat="auto"):
    """
    Convert a string to number
    string:
        <str>, input string
    NaN:
        <int>/<float>, convert "NaN" to certain value
    toFloat:
        "auto"/True/False, set auto to detect "." in the input string automatically
    """
    string = string.strip()
    if "nan" in string.lower():
        return NaN
    elif toFloat == True or toFloat == "auto" and "." in string:
        return float(string)
    elif toFloat == False:
        return int(string)

def find_missing(root, return_type=dict):
    """
    find documents without features in emotion-image output
    Parameter:
        root: the root path of the nested folders
        return_type: dict or list
    Returns:
        a dict or list containing missing documents
    """
    import os
    Missing = {}
    for i, folder in enumerate(filter(lambda x:x[0]!='.', os.listdir(root))):
        files = set(map(lambda x: int(x.split('.')[0]), os.listdir( folder )))
        Missing[folder] = [x for x in range(i*1000,(i+1)*1000) if x not in files]
    
    if return_type == list:
        return reduce(lambda x,y:x+y, Missing.values())
    else:
        return Missing