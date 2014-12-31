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
import hogmodule
from svmmodule import *
import os
#from Tkinter import *
import numpy as np
import cv2
import csv
import caffe

from skimage.feature import hog
from skimage import data, color, exposure
import skimage




                
class caffeDL():
    def __init__(self):
        caffe_root ="/home/lzz/caffe-master/"    
        self.net = caffe.Classifier(caffe_root + 'new/proto/lenet_test.prototxt', caffe_root + 'lenet_iter_3500.caffemodel')
        self.net.set_phase_test()
        self.net.set_mode_cpu()
    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    #net.set_mean('data', np.load(caffe_root + 'mean.binaryproto'))  # ImageNet mean
        self.net.set_raw_scale('data', 1)  # the reference model operates on images in [0,255] range instead of [0,1]
        self.net.set_channel_swap('data', (2,1,0))

class SignWord():
    def __init__(self,path):
        loc=path.rfind("/")
        name=path[loc:]
        self.sampleName=name
        self.wordName=self.sampleName[1:self.sampleName.find(" ")]
        
        self.displacement=[]
        self.handshape=[]
        self.hogFeature=[]
        
        self.keyNo=10
        self.path=path
        self.pred=''
        
        self.label=0
        self.combinedFeature=[]
        
        
        self.loadDisplacement()
        self.getVelocity()
        self.findTopHandshape()
        ''''self.getHogFeature()'''
        print self.wordName
        
        
        
        
        
    def getVelocity(self):
        Label1=open(self.path+"/"+"handshape/velocity"+'.csv','rb')
        vreader = csv.reader(Label1)
        self.vList= []
        self.hList=[]
        self.indexList=[]
        self.value=[]
        for rows in vreader:
            if(os.path.isfile(self.path+"/handshape/"+rows[0]+".jpg") or os.path.isfile(self.path+"/handshape/"+rows[0]+"*.jpg") ): 
                self.indexList.append(rows[0])
                self.vList.append(rows[1])
                self.hList.append(rows[2])
                self.value.append(0)
        for i in range(len(self.indexList)):
            self.value[i]=float(self.hList[i])-10*float(self.vList[i])-0.02*abs(i-len(self.indexList)/2)

        if(len(self.value)<self.keyNo):
            self.keyNo=len(self.value)

        
    def loadDisplacement(self):
        Label1=open(self.path+"/"+'feature.csv','rb')
        reader = csv.reader(Label1)
        for row in reader:
            for x in range (len(row)):
                self.displacement.append(float(row[x]))
        
    def findTopHandshape(self):
        
        self.top=[]
        self.topIndex=[]
        for i in range(self.keyNo):
            self.top.append(self.value[i])
            self.topIndex.append(self.indexList[i])
        #top5Index=[indexList[0],indexList[1],indexList[2],indexList[3],indexList[4]]
        for i in range(self.keyNo,len(self.indexList)):
            if(self.value[i]>min(self.top)):
                ind=self.top.index(min(self.top))
                self.top[ind]=self.value[i]
                self.topIndex[ind]=self.indexList[i]
        #print top5Index
        
        os.chdir(self.path+"/handshape/")
        for i in range(len(self.indexList)):
            if(os.path.isfile(self.path+"/handshape/"+self.indexList[i]+"*.jpg")):
                os.rename(self.indexList[i]+"*.jpg",self.indexList[i]+".jpg")            
        for i in range(len(self.top)):
            os.rename(str(self.topIndex[i])+".jpg",str(self.topIndex[i])+"*.jpg")
        return self.top,self.topIndex




        
        
    def getHogFeature(self):
        files=os.listdir(self.path+"/handshape/")
        hogSet=[]
        for file in files:
            if file[-3:]!="jpg":
                continue
            
            img=cv2.imread(self.path+"/handshape/"+file)
            sp=img.shape
            img2=cv2.copyMakeBorder(img, 0,0, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2), cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
            img3=cv2.resize(img2,(128,128))
            image=img3/255.0
            #image = color.rgb2gray(skimage.data.astronaut())
            image = color.rgb2gray(image)
            
            fd, hog_image = hog(image, orientations=9, pixels_per_cell=(16,16),cells_per_block=(2, 2), visualise=True)
            hogSet.append(fd)

        self.hogFeature=hogmodule.findKey(hogSet)


    def idvdCaffeFeature(self,img_sum,featureTotal,featureTotal2):


        feature=[]
        feature2=[]
        
        for i in range(self.keyNo):
            # print img_sum,i
            feat = featureTotal[img_sum+i]
            feat2 = featureTotal2[img_sum+i]
            #print feat.index(max(feat))
            feature.append(feat)
            feature2.append(feat2)
        #print feature

        self.handshape=self.pooling(feature,1)
        
        self.getVariance(feature2)
        
    
    def pooling(self,feature,types):
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
        
    
    
    
    
    def getVariance(self,feature):
        self.hand_index_list=[]
        for x in range(len(feature)):
            hand_index=feature[x].index(max(feature[x]))
            self.hand_index_list.append(hand_index)
        #hand_result.write(str(l)+" "+index2name[l]+" "+str(hand_index_list)+"\n")
        hand_exist=[]
        #print hand_index_list
        self.variance=0
        for x in range(self.keyNo):

            if((self.hand_index_list[x] in hand_exist)==0):
                hand_exist.append(self.hand_index_list[x])
                self.variance+=1

    
    def combineFeature(self):
        self.combinedFeature=self.displacement+normalize_histogram(self.handshape)+normalize_histogram(self.hogFeature)
        
        
        


        
        
          
class Classifier():
    def __init__(self):
        self.batch_size=80
        self.namepool={}
        self.index2name={}
        self.label=[]
        self.data=[]
        self.displacement=[]
        self.handshape=[]
        self.hogKey=[]
        self.batch=[]
        self.filelist=[]
        self.dic={}
        #data_ind=-1     
        
    def listFile(self,path):
        files = os.listdir(path)
        for file in files:
            if(os.path.isdir(path+"/"+file)==1 and file[0:4]=="HKG_"):
                flag=classifier.testSignWord(path+"/"+file)
                if(flag==0):
                    continue
                self.filelist.append(path+file)
            elif(os.path.isdir(path+"/"+file)==1 and file[0:4]!="HKG_"):
                self.listFile(path+"/"+file)
        return self.filelist
    
    def buildDic(self,path):
        signWord=SignWord(path)
        self.dic[path]=signWord
            
    def imageProcssing(self):
        batch_index=-1
        imgNo=0
        for path in self.filelist:
            for i in range(self.dic[path].keyNo):
                img=cv2.imread(path+"/handshape/"+str(self.dic[path].topIndex[i])+"*.jpg")
                sp=img.shape
                img2=cv2.copyMakeBorder(img, 0,0, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2), cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
                img3=cv2.resize(img2,(128,128))
                img3=img3/255.0
                #imgs.append(img3)
                if(imgNo%self.batch_size==0):
                    self.batch.append([])
                    batch_index+=1
                imgNo+=1
                self.batch[batch_index].append(img3)  
                    
                    
    def constructLabelData(self):
        wordNo=-1
        for path in self.filelist:
            self.dic[path]
            loc=path.rfind("/")
            name=path[loc:]
            word_name=name[0:name.find(" ")]
            if not self.namepool.has_key(word_name):
                word=wordNo+1
                wordNo+=1
                self.namepool[word_name]=word
                self.index2name[path]=name
            else:
                word=self.namepool[word_name]
                self.index2name[path]=name
            self.dic[path].label=word
            self.dic[path].combineFeature()
            #self.label.append(word)           
            #self.displacement.append(self.dic[path].combinedFeature)

        
    
    def getCaffeFeature(self,net):
        self.featureTotal=[]
        self.featureTotal2=[]
        for b in range(len(self.batch)):
            net.predict(self.batch[b],False)
            print "-----------------------------------------------"
            for s in range(len(self.batch[b])):
                feat = net.blobs['ip1'].data[s].flatten().tolist()
                self.featureTotal.append(feat)
                feat2= net.blobs['prob'].data[s].flatten().tolist()
                self.featureTotal2.append(feat2)

    def separateCaffeFeature(self):
        img_sum=0
        for path in self.filelist:
            self.dic[path].idvdCaffeFeature(img_sum,self.featureTotal,self.featureTotal2)
            img_sum+=self.dic[path].keyNo
            
    def testSignWord(self,path):
        files=os.listdir(path+"/handshape/")
        if(len(files)==1):
            return 0
        else:
            return 1
    def split_data(self,classes_in_test):
        trainSet=set()
        testSet=set()
        testLabelSet=set()

        for path in self.filelist:
            if self.dic[path].label  in classes_in_test:
                if ((self.dic[path].label in testLabelSet)==0):
                    testSet.add(path)
                    testLabelSet.add(self.dic[path].label)
                else:
                    trainSet.add(path)
        return trainSet,testSet

    def get_classes_with_at_least_num_of_data(self,num):
            ret=set()
            count={}
            self.label2Name={}
            for path in self.filelist:
                self.label2Name[self.dic[path].label]=self.dic[path].wordName
                
                if not count.has_key(self.dic[path].label):
                    count[self.dic[path].label]=1
                else:
                    count[self.dic[path].label]+=1
            for path in self.filelist:
                if count[self.dic[path].label]>=num:
                    ret.add(self.dic[path].label)
            return ret
        
    def test_svm(self):
        leastNo=2
        intoAccountSet=self.get_classes_with_at_least_num_of_data(leastNo)
        self.trainSet,self.testSet=self.split_data(intoAccountSet)
        train_labels=[]
        train_data=[]
        test_labels=[]
        test_data=[]
        for path in self.trainSet:
            train_labels.append(self.dic[path].label)
            train_data.append(self.dic[path].combinedFeature)
        testpathlist=[]
        for path in self.testSet:
            test_labels.append(self.dic[path].label)
            test_data.append(self.dic[path].combinedFeature)
            testpathlist.append(path)
        
        svm_m1=train_svm_model(train_labels,train_data)
        svm_save_model("/home/lzz/svmModel",svm_m1)
        svm_m2= svm_load_model("/home/lzz/svmModel")
        svm_res1=test_svm_model(svm_m2,test_labels,test_data)
        #plot_a_graph();
        pred_labels=svm_res1[0];
        for i in range (len(testpathlist)):
            path=testpathlist[i]
            self.dic[path].pred=self.label2Name[int(pred_labels[i])]
        
        
        varianceSum1=0
        varianceSum2=0
        same=0
        different=0
        for i in range (len(testpathlist)):
            path=testpathlist[i]
            print self.dic[path].pred,self.dic[path].wordName
            if(self.dic[path].pred==self.dic[path].wordName):
                varianceSum1+=self.dic[path].variance
                same+=1
            else:
                varianceSum2+=self.dic[path].variance
                different+=1
        print float(varianceSum1)/float(same),float(varianceSum2)/float(different)
        return testpathlist
        
    def showResult(self,testpathlist):
        hand_result=open("/home/lzz/project/project/save/Result.txt","w")
        
        for path in self.filelist:
            if path in self.trainSet:
                hand_result.write(self.dic[path].wordName,self.dic[path].hand_index_list)
            if path in self.testSet:
                if self.dic[path].wordName==self.dic[path].pred:
                    TF=1
                else:
                    TF=0
                hand_result.write(TF,self.dic[path].wordName,self.dic[path].pred,self.dic[path].hand_index_list)
    
        hand_result.close()
        

if __name__ == '__main__':
    caffedl=caffeDL()
    classifier = Classifier()
    pathTotal='/media/lzz/Data1/Aaron/1-250/'
    pathTotal='/home/lzz/sample/2data/'
    classifier.listFile(pathTotal)
    #classifier.filelist=["/media/lzz/Data1/Aaron/1-250/HKG_002_a_0002 Aaron 1361","/media/lzz/Data1/Aaron/1-250/HKG_002_a_0002 Aaron 1372","/media/lzz/Data1/Aaron/1-250/HKG_001_a_0001 Aaron 11","/media/lzz/Data1/Aaron/1-250/HKG_001_a_0001 Aaron 22"]
    for path in classifier.filelist:
        classifier.buildDic(path)
    classifier.imageProcssing()
    classifier.getCaffeFeature(caffedl.net)
    classifier.separateCaffeFeature()
    classifier.constructLabelData()
    testPathList=classifier.test_svm()
    classifier.showResult(testPathList)    
    

        


'''class Classifier():
    def __init__(self):
        self.batch_size=80
        self.word=-1
        self.namepool={}
        self.index2name={}
        self.label=[]
        self.data=[]
        self.displacement=[]
        self.handshape=[]
        self.hogKey=[]
        self.data_index=-1
        self.data_index2=-1
        self.lengthList=[]
        self.batch=[]
        self.imgNo=0
        self.batch_index=-1
        self.wordNo=-1
        #data_ind=-1
        
    #print lengthList
    #print "its length",len(lengthList)
        self.variance=[]
        

    def load_velocity(self,path):
        Label1=open(path+"/"+"handshape/velocity"+'.csv','rb')
        vreader = csv.reader(Label1)
        self.vList= []
        self.hList=[]
        self.indexList=[]
        self.value=[]
        self.keyNo=10
        for rows in vreader:
            if(os.path.isfile(path+"/handshape/"+rows[0]+".jpg") or os.path.isfile(path+"/handshape/"+rows[0]+"*.jpg") ): 
                self.indexList.append(rows[0])
                self.vList.append(rows[1])
                self.hList.append(rows[2])
                self.value.append(0)
        for i in range(len(self.indexList)):
            self.value[i]=float(self.hList[i])-10*float(self.vList[i])-0.02*abs(i-len(self.indexList)/2)
        
        if(self.value==[]):
            return []
        if(len(self.value)<self.keyNo):
            self.keyNo=len(self.value)
        self.lengthList.append(self.keyNo)    
        return self.value

    def load_displacement(self,path):
        if(len(self.value)<self.keyNo):
            self.keyNo=len(self.value)
        loc=path.rfind("/")
        name=path[loc:]
        Label1=open(path+"/"+'feature.csv','rb')
        reader = csv.reader(Label1)
        labelArr1 = []
        #data.append([])
        self.data_index+=1
        word_name=name[0:name.find(" ")]
        if not self.namepool.has_key(word_name):
            word=self.wordNo+1
            self.wordNo+=1
            self.namepool[word_name]=word
            self.index2name[self.data_index]=name
        else:
            word=self.namepool[word_name]
            self.index2name[self.data_index]=name
        self.label.append(word)
        
        self.displacement.append([])
        for row in reader:
            for x in range (len(row)):
                self.displacement[self.data_index].append(float(row[x]))

        
        
    def findKey5(self,path):
        
        top5=[]
        top5Index=[]
        for i in range(self.keyNo):
            top5.append(self.value[i])
            top5Index.append(self.indexList[i])
        #top5Index=[indexList[0],indexList[1],indexList[2],indexList[3],indexList[4]]
        for i in range(self.keyNo,len(self.indexList)):
            if(self.value[i]>min(top5)):
                ind=top5.index(min(top5))
                top5[ind]=self.value[i]
                top5Index[ind]=self.indexList[i]
        #print top5Index
        
        os.chdir(path+"/handshape/")
        for i in range(len(self.indexList)):
            if(os.path.isfile(path+"/handshape/"+self.indexList[i]+"*.jpg")):
                os.rename(self.indexList[i]+"*.jpg",self.indexList[i]+".jpg")            
        for i in range(len(top5)):
            os.rename(str(top5Index[i])+".jpg",str(top5Index[i])+"*.jpg")
        return top5,top5Index




    def imageProcssing(self,path,top5,top5Index):
        for i in range(len(top5)):
            img=cv2.imread(path+"/handshape/"+str(top5Index[i])+"*.jpg")
            sp=img.shape
            img2=cv2.copyMakeBorder(img, 0,0, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2), cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
            img3=cv2.resize(img2,(128,128))
            img3=img3/255.0
            #imgs.append(img3)
            if(self.imgNo%self.batch_size==0):
                self.batch.append([])
                self.batch_index+=1
            self.imgNo+=1
            self.batch[self.batch_index].append(img3)
            for i in range(len(self.batch)-1):
                assert len(self.batch[i])==self.batch_size

    def getHogFeature(self,path):
        self.data_index2+=1
        files=os.listdir(path+"/handshape/")
        hogSet=[]
        for file in files:
            if file[-3:]!="jpg":
                continue
            
            img=cv2.imread(path+"/handshape/"+file)
            sp=img.shape
            img2=cv2.copyMakeBorder(img, 0,0, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2), cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
            img3=cv2.resize(img2,(128,128))
            image=img3/255.0
            #image = color.rgb2gray(skimage.data.astronaut())
            image = color.rgb2gray(image)
            
            fd, hog_image = hog(image, orientations=9, pixels_per_cell=(16,16),cells_per_block=(2, 2), visualise=True)
            hogSet.append(fd)

        hogFeature=hogmodule.findKey(hogSet)

        self.hogKey.append([])
        self.hogKey[self.data_index2]=hogFeature

        return hogFeature

        
        
        

    def getCaffeFeature(self,net):
        self.featureTotal=[]
        self.featureTotal2=[]
        for b in range(len(self.batch)):
            net.predict(self.batch[b],False)
            print "-----------------------------------------------"
            for s in range(len(self.batch[b])):
                feat = net.blobs['ip1'].data[s].flatten().tolist()
                self.featureTotal.append(feat)
                feat2= net.blobs['prob'].data[s].flatten().tolist()
                self.featureTotal2.append(feat2)

    def separateCaffeFeature(self):
        img_sum=0
        data_ind=-1
        handIndexList=[]
        for l in range(len(self.lengthList)):
            data_ind+=1
            hand_index_list=self.idvdCaffeFeature(l,img_sum,data_ind)
            img_sum+=self.lengthList[l]
            handIndexList.append(hand_index_list)
        return handIndexList
            
        
    def idvdCaffeFeature(self,l,img_sum,data_ind):


        feature=[]
        feature2=[]
        
        for i in range(self.lengthList[l]):
            # print img_sum,i
            feat = self.featureTotal[img_sum+i]
            feat2 = self.featureTotal2[img_sum+i]
            #print feat.index(max(feat))
            feature.append(feat)
            feature2.append(feat2)
        #print feature

        self.handshape.append([])
        self.handshape[data_ind]=self.pooling(feature,1)
        
        hand_index_list=self.getVariance(feature2,l)
        
    
    def pooling(self,feature,types):
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
        
    
    
    
    
    def getVariance(self,feature,l):
        hand_index_list=[]
        for x in range(len(feature)):
            hand_index=feature[x].index(max(feature[x]))
            hand_index_list.append(hand_index)
        #hand_result.write(str(l)+" "+index2name[l]+" "+str(hand_index_list)+"\n")
        hand_exist=[]
        #print hand_index_list
        self.variance.append(0)
        for x in range(len(hand_index_list)):

            if((hand_index_list[x] in hand_exist)==0):
                hand_exist.append(hand_index_list[x])
                self.variance[l]+=1
        return hand_index_list
    
    
    def combination(self,displacement,hogKey,handshape):
        assert len(displacement)==len(hogKey)
        for l in range(len(self.lengthList)):
            self.data.append([])
            self.data[l]=displacement[l]+normalize_histogram(hogKey[l])+normalize_histogram(handshape[l])

    def showResult(self,predLabel,testLabel,initial_index_test,testClass2initialIndex):
        hand_result=open("/home/lzz/project/project/save/Result.txt","w")
        for i in range(len(predLabel)):
            if testLabel[i]==predLabel[i]:
                TF=1
            else:
                TF=0
            hand_result.write(TF,classifier.index2name[testClass2initialIndex[testLabel[i]]],classifier.index2name[testClass2initialIndex[predLabel[i]]],handIndexList[testClass2initialIndex[testLabel[i]]],classifier.variance[initial_index_test[i]])
        hand_result.close()
        
if __name__ == '__main__':
    caffedl=caffeDL()
    classifier = Classifier()
    path='/media/lzz/Data1/Aaron/1-250/'
    filelist=[]
    filelist=list_file(path,filelist)
    filelist=["/media/lzz/Data1/Aaron/1-250/HKG_002_a_0002 Aaron 1361","/media/lzz/Data1/Aaron/1-250/HKG_002_a_0002 Aaron 1372","/media/lzz/Data1/Aaron/1-250/HKG_001_a_0001 Aaron 11","/media/lzz/Data1/Aaron/1-250/HKG_001_a_0001 Aaron 22"]
    for path in filelist:
        print path
        if(classifier.load_velocity(path)==[]):
            continue
        classifier.load_displacement(path)
        classifier.getHogFeature(path)
        top5,top5Index=classifier.findKey5(path)
        classifier.imageProcssing(path,top5,top5Index)
    
        
    classifier.getCaffeFeature(caffedl.net)
    
    handIndexList=classifier.separateCaffeFeature()
      
    
    
    classifier.combination(classifier.displacement,classifier.hogKey,classifier.handshape)
    assert len(classifier.label)==len(classifier.data)
    predLabel,testLabel,initial_index_test,testClass2initialIndex=test_svm(classifier.label,classifier.data,classifier.variance)
    classifier.showResult(predLabel,testLabel,initial_index_test,testClass2initialIndex)'''
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
'''if __name__ == '__main__':
    
    path='/media/lzz/Data1/Aaron/1-250/'
    filelist=[]
    filelist=list_file(path,filelist)
    
    
    #filelist=["/media/lzz/Data1/Aaron/1-250/HKG_002_a_0002 Aaron 1361","/media/lzz/Data1/Aaron/1-250/HKG_002_a_0002 Aaron 1372","/media/lzz/Data1/Aaron/1-250/HKG_001_a_0001 Aaron 11","/media/lzz/Data1/Aaron/1-250/HKG_001_a_0001 Aaron 22"]
    #try:
    word=-1##################################################
    namepool={}
    index2name={}
    label=[]
    data=[]
    displacement=[]
    handshape=[]
    data_index=-1

    batch_size=80#################################################
    caffe_root ="/home/lzz/caffe-master/"    
    net = caffe.Classifier(caffe_root + 'new/proto/lenet_test.prototxt', caffe_root + 'lenet_iter_3500.caffemodel')
    net.set_phase_test()
    net.set_mode_cpu()
# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
#net.set_mean('data', np.load(caffe_root + 'mean.binaryproto'))  # ImageNet mean
    net.set_raw_scale('data', 1)  # the reference model operates on images in [0,255] range instead of [0,1]
    net.set_channel_swap('data', (2,1,0))
    
    lengthList=[]################################################
    batch=[]
    imgNo=0
    batch_index=-1
    wordNo=-1
    print len(filelist)
    
    
    
    for path in filelist:
        
            Label1=open(path+"/"+"handshape/velocity"+'.csv','rb')#####################
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
                    
                    
                    
                    
            if(value==[]):****************
                continue
                
                
                
                
                
                
                
                
                
                
            if(len(value)<keyNo):############
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
            
            

            
            for i in range(len(indexList)):###########
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
    test_svm(label,data,variance)'''