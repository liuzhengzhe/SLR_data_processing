#import matplotlib.pyplot as plt
#from matplotlib.pyplot import savefig




import Queue

import json
import gc
import glob
from dtw import dtw
import re
drname="handshape"
from Classifier import *
import sys
sys.path.insert(0,'/home/lzz/project/project/lstm/')

import lstm.RNN_with_gating

#import whole_network,whole_level_network

if __name__ == '__main__':
    #caffedl=caffeDL('/media/lzz/65c50da0-a3a2-4117-8a72-7b37fd81b574/sign/proto/lenet_test.prototxt','/media/lzz/65c50da0-a3a2-4117-8a72-7b37fd81b574/sign/model/lenet_iter_5000.caffemodel')
    caffedl=caffeDL('/home/lzz/caffe/caffe-master/examples/imagenet/train_val_16_py.prototxt','/home/lzz/caffe/caffe-master/examples/imagenet/model/4096_iter_10000.caffemodel')
    #caffedlInter=caffeDL('/media/lzz/65c50da0-a3a2-4117-8a72-7b37fd81b574/sign/proto_inter/lenet_test.prototxt','/media/lzz/65c50da0-a3a2-4117-8a72-7b37fd81b574/sign/model/lenet__iter_400.caffemodel')
    caffedlInter=caffeDL('/home/lzz/caffe/caffe-master/examples/imagenet/intermodel/train_val_inter.prototxt','/home/lzz/caffe/caffe-master/examples/imagenet/intermodel/24inter_iter_300.caffemodel')
    classifier = Classifier()
    #pathTotal='/media/lzz/HD1/1Michael/split/301-610new/'
    #pathTotal='/media/lzz/HD1/1Michael/split/791-1000/'
    #pathTotal='/media/lzz/HD1/1Michael/split/1-23/'
    #pathTotal='/media/lzz/HD1/1Michael/split/new/'
    #pathTotal='/media/lzz/HD1/1Michael/split/new/301-610new/'
    #pathTotal='/media/lzz/HD1/1Michael/split/new/1-250/'
    #pathTotal='/media/lzz/HD1/1Michael/split/new/1-250/Aaron 1-180/'
    #pathTotal='/media/lzz/Data1/michael/301-400/'
    #pathTotal="/home/lzz/hand/"
    #pathTotal='/media/lzz/HD1/real/'
    #pathTotal='/home/lzz/sign/data/'
    #pathTotal='/media/lzz/Data1/kinect/'
    #pathTotal='/home/lzz//sign/data/'


    dataset='our'

    trainname={}
    testname={}

    if dataset=='devisign':
        #pathTotal='/media/lzz/Data1/devisign/'
        #pathTotal='/media/lzz/Data1/own/'
        pathTotal='/home/lzz/sign/data0/try/'
        trainname['P08']=0
        trainname['P02']=0
        trainname['P01']=0
        trainname['P07']=0
        #trainname['P01']=0
        testname['P03']=0
        classifier.listdevisign(pathTotal)
        classifier.splitdevisign(trainname,testname,'',0)
        #classifier.splitdevisign(trainname,testname,'P01',1)
    elif dataset=='our':
        #pathTotal='/home/lzz/sign/data/'
        pathTotal='/media/lzz/Data1/kinect/try/'
        #pathTotal='/media/lzz/HD1/newkinect/'
        trainname['hfy']=0
        trainname['fuyang']=0

        trainname['lzz']=0
        testname['Aaron']=0
        testname['Michael']=0
        testname['Micheal']=0
        #classifier.listFile(pathTotal)
        #classifier.split(trainname,testname,'xn',3)
        #classifier.split(trainname,testname,'lzz',1)
        dic={}
        dic['Aaron']=0
        dic['Michael']=0
        dic['Micheal']=0
        classifier.listFile(pathTotal)
        print 'finish list'
        classifier.split(trainname,testname,dic,0)

    print 'finish initialization.'
    classifier.constructLabelData()
    classifier.label2Name={}
    for path in classifier.filelist:
        classifier.label2Name[classifier.dic[path].label]=classifier.dic[path].wordName

    #print 'copying...'

    '''list=open('./images.txt','w')
    classifier.enlarge(list)
    list.close()'''



    #for path in classifier.filelist:
    #    classifier.buildDic(path,0)

    mode=2


    #1:test;2:save model
    wholeMode=1


    classifier.getBothSeparate()
    if mode==2:
        classifier.getInter()








    classifier.getVelo()
    #for f in classifier.filelist:
    #    print classifier.dic[f].path,classifier.dic[f].intersect,classifier.dic[f].bothseparate,classifier.dic[f].shouldgenerate
    classifier.checkType()

    for path in classifier.filelist:
        txt=open(classifier.dic[path].path+"/type.txt",'w')
        if classifier.dic[path].intersect==1 and classifier.dic[path].shouldgenerate==0:
            txt.write("intersect")
        if classifier.dic[path].bothseparate==1:
            txt.write("both")
        if classifier.dic[path].shouldgenerate==1:
            txt.write("generated")
        txt.close()

    classifier.getVelocity()

    classifier.findTopHandshape()

    classifier.constructTrajectory()

    classifier.getHogFeature()


    #[[train1,train2,train3],[target1,target2,target3],[t1,t2,t3]],[[test1,test2,test3],[testtarget1,testtarget2,testtarget3],[t1,t2,t3]]=classifier.preLSTM()
    #classifier.trainWholeCombineFeature()

    #classifier.trainWhole()
    #classifier.prewhole()
    #classifier.trainWholeLevelFeature()

    #classifier.trainLSTM()
    #classifier.testLSTM(test1,test2,test3,testtarget1,testtarget2,testtarget3,t1,t2,t3)
    classifier.getCaffeFeature(caffedl.net,caffedlInter.net)

    classifier.separateCaffeFeature()

    #if wholeMode==2:
    #    classifier.makeXml()


    classifier.signkey()
    w1=1
    w2=1
    w3=1
    classifier.combineFeature(w1,w2,w3)
    w1=1
    w2=1
    w3=1
    try:
        classifier.getDifficulty()
    except:
        pass
    print 'loading..'
    #classifier.loadfeature(w1,w2,w3)

    #classifier.savehdf5()
    try:
        classifier.trajehdf5()
    except:
        pass
    classifier.test_svm(w1,w2,w3)



