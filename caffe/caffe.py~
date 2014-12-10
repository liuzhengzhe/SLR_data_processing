'''
Created on Oct 24, 2014

@author: liuzz
'''
from svmutil import *
from svmmodule import *
labels=[1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5]
feat=open('D:/eclipse/project/save/generating/caffe/try.txt')
fe=feat.read().split(" ")
feature=[]
for i in range(len(fe)-1):
    fe[i]=float(fe[i])
for i in range(20):
    feature.append([])
for i in range(20):
    label=int(fe[i*4097]);
    for j in range(4096):
        feature[label].append(fe[i*4097+1+j])

test_svm(labels,feature,bin_num=4,level_num=2,level_num_hog=3,para='-s 0 -c 2048 -t 2 -g 0.5')
