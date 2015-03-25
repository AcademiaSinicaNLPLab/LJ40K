# -*- coding: utf-8 -*-
import sys, os
sys.path.append("../")
import random
import pickle
from feelit.utils import LJ40K

def help():
    print
    print "usage: python %s [Data_quantity][Selected_EachEm_Data_quantity][train/test]"  % (__file__)
    print
    print "-----------------------------------------------------------------------------------"
    print "  e.g: python %s 32000 80 train" % (__file__)
    print "  from 32000 training data randomly select 80*2(80pos., 80neg.) data each emotion for training"
    print "  e.g: output pkl file: {'accomplished':[3,56,34,78...],'sad':[466,536,423,...],.....}"
    print
    print "-----------------------------------------------------------------------------------"
    print "  e.g: python %s 8000 20 test" % (__file__)
    print "  from 8000 testing data randomly select 20 data each emotion for testing"
    print
    exit(-1)

def random_positive_label_data(Data_quantity,EachEm_Data_quantity,Selected_EachEm_Data_quantity):
    selected_Data = []
    #build[0,1,2,....31999]
    feature_indexs = range(Data_quantity)
    #build[0,800,1600,....31200] and make feature_indexs separated 
    for i in range(0,Data_quantity,EachEm_Data_quantity):
        #build[0,1,2,3,....,799]
        Part_of_feature_index = feature_indexs[i:i+EachEm_Data_quantity]
        random.shuffle(Part_of_feature_index)
        selected_Data.append(Part_of_feature_index[:Selected_EachEm_Data_quantity])
    return selected_Data

def random_negative_label_data(Data_quantity,EachEm_Data_quantity,Selected_EachEm_Data_quantity):    
    selected_Data = []
    #build[0,800,1600,....31200] and make feature_indexs separated 
    for i in range(0,Data_quantity,EachEm_Data_quantity):
        feature_indexs = range(Data_quantity)
        # delete index of positive label
        del feature_indexs[i:i+EachEm_Data_quantity]
        random.shuffle(feature_indexs)
        feature_indexs = feature_indexs[:Selected_EachEm_Data_quantity]
        selected_Data.append(feature_indexs)
    # print selected_Data
    return selected_Data

def random_data(Data_quantity,Emotion_quantity,Selected_EachEm_Data_quantity):     
    selected_Data = []
    #build[0,200,400,....7800] and make feature_indexs separated      
    for i in range(Emotion_quantity):
        #build[0,1,2,....7999]
        feature_indexs = range(Data_quantity)
        random.shuffle(feature_indexs)
        selected_Data.append(feature_indexs[:Selected_EachEm_Data_quantity])
    return selected_Data

if __name__ == '__main__':

    if len(sys.argv) != 4: help()
    Emotion_quantity = len(LJ40K)
    Data_quantity = int(sys.argv[1])
    #EachEm_Data_quantity = 32000/40 = 800
    EachEm_Data_quantity = Data_quantity/Emotion_quantity
    # e.g: Selected_EachEm_Data_quantity = 80
    Selected_EachEm_Data_quantity = int(sys.argv[2])
    
    if sys.argv[3] == 'train':
        SelectPartP = random_positive_label_data(Data_quantity,EachEm_Data_quantity,Selected_EachEm_Data_quantity)
        SelectPartN = random_negative_label_data(Data_quantity,EachEm_Data_quantity,Selected_EachEm_Data_quantity)

        Total_selecting_Data = [x+y for x, y in zip(SelectPartP, SelectPartN)]
        SETQ = str(Selected_EachEm_Data_quantity)
        random_idx = dict(zip(LJ40K,Total_selecting_Data))
        pickle.dump(random_idx, open("random"+SETQ+"p"+SETQ+"ntrain_idx.pkl", "wb"), protocol=2)
    
    elif sys.argv[3] == 'test':
        Total_selecting_Data = random_data(Data_quantity,Emotion_quantity,Selected_EachEm_Data_quantity)
        SETQ = str(Selected_EachEm_Data_quantity)
        random_idx = dict(zip(LJ40K,Total_selecting_Data))
        pickle.dump(random_idx, open("random"+SETQ+"test_idx.pkl", "wb"), protocol=2)
        
    else:
        help()
