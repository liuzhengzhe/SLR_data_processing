'''

@author: qxj
'''
from svmutil import *
from svmmodule import *
import matplotlib
import sqlite3
import math
import numpy
import struct
#from hmm.continuous.GMHMM import GMHMM
#from cluster import KMeansClustering
import time
import random
import matplotlib.pyplot as plt
import marshal, pickle
import svmmodule
from hmmmodule import *
from basic import *
from load import *
from constant_numbers import *
from hodmodule import *
from hog_template import *
from svmmodule import *
import os
#from Tkinter import *
import numpy as np
import cv2
import csv



label2classNo={}

def load_data_no_mog(path0,filelist):
    '''files = os.listdir(path)
    for f in files:
        if os.path.isdir(path+f)==0:
            continue'''
    current_data_index=-1
    data=[]
    label=[]
    namelist=[]
    read_index=[1,3,4,5,6,7,8,9]
    data_index=-1
    namepool={}
    word=-1
    for path in filelist:
        current_data_index+=1
        loc=path.rfind("/")
        name=path[loc:]
        Label1=open(path+"/"+name+'.csv','rb')
        reader = csv.reader(Label1)
        labelArr1 = []
        data.append([])
        data_index+=1
        word_name=name[0:name.find(" ")]
        if not namepool.has_key(word_name):
            word+=1
            namepool[word_name]=word
        else:
            word=namepool[word_name]
        label.append(word)
        namelist.append(name)
        for row in reader:
            if(row[0]!="untracked"):
                labelArr1.append(row)
                row2=[]
                for index in read_index:
                    row2.append(float(row[index*7]))
                    row2.append(float(row[index*7+1]))
                    row2.append(float(row[index*7+2]))
                data[data_index].append(row2)
        assert(len(label)==len(data))
    return label,data,namelist
    
    


def construct_features(labels,data,namelist,bin_num=8,level_num=2,level_num_hog=3):
    assert len(labels)==len(data)
    #templates=load_templates("../data/hog_60template15mean.txt")
    #templates=load_templates("F:/study/save/generating/handshape/tmp.txt")
    ret_labels=[]
    features=[]

    for i in range(len(labels)):
        frames=data[i]
        if(i==127):
            i=127      
        if len(frames)==0:
            print i,labels[i],frames
            continue
        movement=hod(frames,bin_num,level_num)
        movement_descriptor=movement[0]
#        deter_left=movement[1]
#        deter_right=movement[2]
    
        #hand_shape_descriptor=construct_hog_binary_features(frames,templates,1,level_num_hog,deter_left,deter_right)
#        if sum(hand_shape_descriptor)==0:
#            print i,'no length'
#            continue
        ret_labels.append(labels[i])
        features.append(normalize_histogram(movement_descriptor))
        #features.append(normalize_histogram(movement_descriptor)+normalize_histogram(hand_shape_descriptor))

    return ret_labels,features



    
def list_file(path,filelist):
    files = os.listdir(path)
    for file in files:
        if(os.path.isdir(path+"/"+file)==1 and file[0:4]=="HKG_"):
            filelist.append(path+file)
        elif(os.path.isdir(path+"/"+file)==1 and file[0:4]!="HKG_"):
            list_file(path+"/"+file,filelist)
    return filelist
 



if __name__ == '__main__':
    #path='H:/Aaron/1-250/'
    path='/media/lzz/Data1/Aaron/1-250/'
    filelist=[]
    filelist=list_file(path,filelist)
    
    
    
    #try:
    word=-1
    namepool={}
    label=[]
    data=[]
    data_index=-1
    for path in filelist:
            loc=path.rfind("/")
            name=path[loc:]
            Label1=open(path+"/"+'feature.csv','rb')
            reader = csv.reader(Label1)
            labelArr1 = []
            data.append([])
            data_index+=1
            word_name=name[0:name.find(" ")]
            if not namepool.has_key(word_name):
                word+=1
                namepool[word_name]=word
            else:
                word=namepool[word_name]
            label.append(word)
            for row in reader:
                for i in range(len(row)):
                    data[data_index].append(float(row[i]))
    test_svm(label,data)
    '''except:
        [label,data,name]=load_data_no_mog(path,filelist)
    
        [labels,features,name]=construct_features(label,data,name)
    
        for l in range(len(labels)): 
            csvfile = file(path+name[l]+'/feature.csv', 'wb')
            writer = csv.writer(csvfile)

            data=[(features[l])]

            writer.writerows(data)

            csvfile.close()'''
