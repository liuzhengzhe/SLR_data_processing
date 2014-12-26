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
#from hmmmodule import *
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
import caffe

def list_file(path,filelist):
    files = os.listdir(path)
    for file in files:
        if(os.path.isdir(path+"/"+file)==1 and file[0:4]=="HKG_"):
            filelist.append(path+file)
        elif(os.path.isdir(path+"/"+file)==1 and file[0:4]!="HKG_"):
            list_file(path+"/"+file,filelist)
    return filelist

def pooling(feature,types):
    handshape=[]
    if(types==0):
        #max pooling
        for i in range(len(feature[0])):
            maxvalue=feature[0][i]
            for j in range(len(feature)):
                if(feature[j][i]>maxvalue):
                    maxvalue=feature[j][i]
            handshape.append(maxvalue)
        return handshape
    if(types==1):
        for i in range(len(feature[0])):
            sum0=0
            for j in range(len(feature)):
                sum0=sum0+feature[j][i]
            ave=sum0/len(feature)
            handshape.append(ave)
        return handshape
    if(types==2):
        for i in range(len(feature[0])):
            seq=[]
            for j in range(len(feature)):
                seq.append(feature[j][i])
            seq1=sorted(seq)
            midValue=seq1[len(seq1) // 2]
            handshape.append(midValue)
        return handshape
                

if __name__ == '__main__':
    
    path='/media/lzz/Data1/Aaron/1-250/'
    filelist=[]
    filelist=list_file(path,filelist)
    
    
    #filelist=["/media/lzz/Data1/Aaron/1-250/HKG_002_a_0002 Aaron 1361","/media/lzz/Data1/Aaron/1-250/HKG_002_a_0002 Aaron 1372","/media/lzz/Data1/Aaron/1-250/HKG_001_a_0001 Aaron 11","/media/lzz/Data1/Aaron/1-250/HKG_001_a_0001 Aaron 22"]
    #try:
    word=-1
    namepool={}
    index2name={}
    label=[]
    data=[]
    displacement=[]
    handshape=[]
    data_index=-1

    batch_size=80
    caffe_root ="/home/lzz/caffe-master/"    
    net = caffe.Classifier(caffe_root + 'new/proto/lenet_test.prototxt', caffe_root + 'lenet_iter_3500.caffemodel')
    net.set_phase_test()
    net.set_mode_cpu()
# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
#net.set_mean('data', np.load(caffe_root + 'mean.binaryproto'))  # ImageNet mean
    net.set_raw_scale('data', 1)  # the reference model operates on images in [0,255] range instead of [0,1]
    net.set_channel_swap('data', (2,1,0))
    
    lengthList=[]
    batch=[]
    imgNo=0
    batch_index=-1
    wordNo=-1
    print len(filelist)
    for path in filelist:
        
            Label1=open(path+"/"+"handshape/velocity"+'.csv','rb')
            vreader = csv.reader(Label1)
            vList= []
            hList=[]
            indexList=[]
            value=[]
            keyNo=10
            for rows in vreader:
                if(os.path.isfile(path+"/handshape/"+rows[0]+".jpg") or os.path.isfile(path+"/handshape/"+rows[0]+"*.jpg") ): 
                    indexList.append(rows[0])
                    vList.append(rows[1])
                    hList.append(rows[2])
                    value.append(0)
            if(value==[]):
                continue
            if(len(value)<keyNo):
                keyNo=len(value)
            loc=path.rfind("/")
            name=path[loc:]
            Label1=open(path+"/"+'feature.csv','rb')
            reader = csv.reader(Label1)
            labelArr1 = []
            #data.append([])
            data_index+=1
            word_name=name[0:name.find(" ")]
            if not namepool.has_key(word_name):
                word=wordNo+1
                wordNo+=1
                namepool[word_name]=word
                index2name[data_index]=name
            else:
                word=namepool[word_name]
                index2name[data_index]=name
            label.append(word)
            
            displacement.append([])
            for row in reader:
                for x in range (len(row)):
                    displacement[data_index].append(float(row[x]))
            
            

            
            for i in range(len(indexList)):
                value[i]=float(hList[i])-10*float(vList[i])-0.02*abs(i-len(indexList)/2)
                #print indexList[i],value[i]
            print path
            #top5=[value[0],value[1],value[2],value[3],value[4]]

            lengthList.append(keyNo)
            top5=[]
            top5Index=[]
            for i in range(keyNo):
                top5.append(value[i])
                top5Index.append(indexList[i])
            #top5Index=[indexList[0],indexList[1],indexList[2],indexList[3],indexList[4]]
            for i in range(keyNo,len(indexList)):
                if(value[i]>min(top5)):
                    ind=top5.index(min(top5))
                    top5[ind]=value[i]
                    top5Index[ind]=indexList[i]
            #print top5Index
            
            os.chdir(path+"/handshape/")
            for i in range(len(indexList)):
                if(os.path.isfile(path+"/handshape/"+indexList[i]+"*.jpg")):
                    os.rename(indexList[i]+"*.jpg",indexList[i]+".jpg")            
            for i in range(len(top5)):
                os.rename(str(top5Index[i])+".jpg",str(top5Index[i])+"*.jpg")
                
            
            
            
            for i in range(len(top5)):
                img=cv2.imread(path+"/handshape/"+str(top5Index[i])+"*.jpg")
                sp=img.shape
                img2=cv2.copyMakeBorder(img, 0,0, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2), cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
                img3=cv2.resize(img2,(128,128))
                img3=img3/255.0
                #imgs.append(img3)
                if(imgNo%batch_size==0):
                    batch.append([])
                    batch_index+=1
                imgNo+=1
                batch[batch_index].append(img3)
                for i in range(len(batch)-1):
                    assert len(batch[i])==batch_size
    print imgNo
    
    labellist=[]
    for i in range(len(label)):
        labellist.append(0)
    for i in range(len(label)):
        labellist[label[i]]+=1

        

    
    
    
    featureTotal=[]
    for b in range(len(batch)):
        net.predict(batch[b],False)
        print "-----------------------------------------------"
        for s in range(len(batch[b])):
            feat = net.blobs['ip1'].data[s].flatten().tolist()
            featureTotal.append(feat)
        
    #print "featureTotal=",len(featureTotal)
    data_ind=-1
    img_sum=0
    #print lengthList
    #print "its length",len(lengthList)
    variance=[]
    #hand_result=open("/home/lzz/project/project/save/hand_result.txt","w")
    
    for l in range(len(lengthList)):
        data_ind+=1
        feature=[]
       
        for i in range(lengthList[l]):
            # print img_sum,i
            feat = featureTotal[img_sum+i]

            #print feat.index(max(feat))
            feature.append(feat)
        img_sum+=lengthList[l]
        handshape.append([])
        handshape[data_ind]=pooling(feature,1)
        
        
        
        hand_index_list=[]
        for x in range(len(feature)):
            hand_index=feature[x].index(max(feature[x]))
            hand_index_list.append(hand_index)
        #hand_result.write(str(l)+" "+index2name[l]+" "+str(hand_index_list)+"\n")
        hand_exist=[]

        variance.append(0)
        for x in range(len(hand_index_list)):

            if((hand_index_list[x] in hand_exist)==0):
                hand_exist.append(hand_index_list[x])
                variance[l]+=1
        
        
        data.append([])
        #data[data_ind]=normalize_histogram(handshape[data_ind])
        data[data_ind]=displacement[data_ind]+normalize_histogram(handshape[data_ind])
        #data[data_ind]=displacement[data_ind]+normalize_histogram(handshape[data_ind])
            #print len(data[data_index])
    #print img_sum
    #print len(data[0])
       
        
    assert len(label)==len(data)
    test_svm(label,data,variance)