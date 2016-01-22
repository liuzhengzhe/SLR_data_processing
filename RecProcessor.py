from svmutil import *
from svmmodule import *
import random
import time
import binascii
import bz2
import datetime
from FrameConverter import FrameConverter
import collections
import pylab as plt
import cv2
#from caffeDL import *

#import slWord
import csv
import asyncore
import Queue
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
import shutil
import thread
import threading
import sys, traceback
from Main_Test import *
class RecProcessor(object):

    def __init__(self,send_data_callback):
        shutil.rmtree('/media/lzz/65c50da0-a3a2-4117-8a72-7b37fd81b574/sign/realtime/handshape')
        shutil.rmtree('/media/lzz/65c50da0-a3a2-4117-8a72-7b37fd81b574/sign/realtime/rawshape')
        os.mkdir('/media/lzz/65c50da0-a3a2-4117-8a72-7b37fd81b574/sign/realtime/handshape')
        os.mkdir('/media/lzz/65c50da0-a3a2-4117-8a72-7b37fd81b574/sign/realtime/rawshape')
        self.fr=0
        self.one=0
        self.exist=0
        self.lock = threading.Lock()
        self.condition=threading.Condition()
        self.classifier=Classifier()
        self.classifier.filelist.append('/home/lzz/sign/svm/')
        self.classifier.dic['/home/lzz/sign/svm/']=SignWord('/home/lzz/sign/svm/',0)
        self.classifier.dic['/home/lzz/sign/svm/'].skeletons=[]
        self.classifier.dic['/home/lzz/sign/svm/'].positions=[]
        self.send_data=send_data_callback

        self.converter = FrameConverter()
        if hasattr(RecProcessor,'caffedl'):
            RecProcessor.caffedl=caffeDL('/home/lzz/caffe/caffe-master/examples/imagenet/train_val_16_py_rec.prototxt','/home/lzz/caffe/caffe-master/examples/imagenet/model/4096_iter_10000.caffemodel')
        if hasattr(RecProcessor,'caffedlInter'):
            RecProcessor.caffedlInter=caffeDL('/home/lzz/caffe/caffe-master/examples/imagenet/train_val_16_py_rec.prototxt','/home/lzz/caffe/caffe-master/examples/imagenet/intermodel/24inter_iter_300.caffemodel')
        self.signDic={}
        self.queue = Queue.Queue(maxsize = 10000)
        Label1=open('/media/lzz/65c50da0-a3a2-4117-8a72-7b37fd81b574/sign/svm/dictionary.csv','rb')
        reader = csv.reader(Label1)
        for row in reader:
            self.signDic[int(row[0])]=row[1]
        #self.server = EchoServer(port,  self.received_data)
        thread.start_new_thread(self.threadfunc,())
        #asyncore.loop()

    def received_data(self, received_data):
        #print (len(received_data), binascii.crc32(received_data))
        #decompressed_data = bz2.decompress(received_data)
        decoded_data = self.converter.decode(received_data)
        return self.process_data(decoded_data)


    '''def process_data_handshape(self, decoded_data):
        if(self.queue.full()==0):
            self.condition.acquire()
            self.queue.put(decoded_data)
            self.condition.notify()
            self.condition.release()
        else:
            self.server.send("full"+"#TERMINATOR#")

    def threadfunc_handshape(self):
        while(1):
            if(self.queue.empty()==1):
                self.condition.acquire()
                self.condition.wait()
                self.condition.release()
            decoded_data=self.queue.get()
            img=decoded_data['right']
            sp=img.shape
            if sp[0]>sp[1]:
                img2=cv2.copyMakeBorder(img, 0,0, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2), cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
            else:
                img2=cv2.copyMakeBorder(img, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2),0,0, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
            img3=cv2.resize(img2,(128,128))
            img3=img3/255.0
            self.classifier.batch.append(img3)
            self.caffedl.net.predict(self.classifier.batch,False)

            feat = self.caffedl.net.blobs['prob'].data[0].flatten().tolist()
            maxi=0
            result=-1
            for i in range(len(feat)):
                if maxi<feat[i]:
                    maxi=feat[i]
                    result=i
            print result

            self.server.handler.send(str(result)+'#TERMINATOR#')'''


    def process_data(self, decoded_data):
        #print decoded_data
        if(self.queue.full()==0):
            if self.exist==0:
                if decoded_data["label"]=='End':
                    if self.lock.acquire():
                        self.exist=1#exist means there is a End in the queue, that is, a complete sign exists
                        self.lock.release()
            else:
                if self.lock.acquire():
                    self.queue.queue.clear()
                    self.classifier.buffer=[]
                    self.exist=0
                    self.lock.release()
                    '''timedata={}
                    timedata['label']='time'
                    timedata['position']=datetime.datetime.now()
                    #print "starttime",timedata['position']
                    self.queue.put(timedata)'''
            self.queue.put(decoded_data)
            self.condition.acquire()
            self.condition.notify()
            self.condition.release()
        else:
            self.send_data("Redo"+'#TERMINATOR#')
            if self.lock.acquire():
                self.queue.queue.clear()
                self.classifier.buffer=[]
                self.exist=0
                self.lock.release()
        return "0"

    def threadfunc(self):
        while(1):
            self.condition.acquire()
            if(self.queue.empty()==1):
                self.condition.wait()
                self.condition.release()
            decoded_data=self.queue.get()
            if self.lock.acquire():
                self.classifier.buffer.append(decoded_data)
                self.lock.release()
            if(decoded_data["label"]=="End"):
                #print len(self.classifier.buffer)
                if self.lock.acquire():
                    self.exist=0
                    self.lock.release()
                if len(self.classifier.buffer)<10:
                    self.send_data("Redo"+"#TERMINATOR#")
                else:
                    self.process_all()
                if self.lock.acquire():
                    self.classifier.buffer=[]
                    self.lock.release()





                    '''starttimedic=self.queue.get()
                    starttime=starttimedic['position']
                    endtime = datetime.datetime.now()
                    print "endtime",endtime
                    print (endtime - starttime).seconds
                    if (endtime - starttime).seconds<60 and len(self.classifier.buffer)>10:'''
    def ret(self):
        self.fr=0
        self.one=0

        self.classifier.reset()


    def process_all(self):

        path="/home/lzz/sign/svm/"
        buf=self.classifier.buffer
        for decoded_data in buf:
            if decoded_data["label"]=="time":
                self.ret()
                return
            if(decoded_data["label"]=="End"):
                break
            read_index=[1,3,4,5,6,7,8,9]
            raw=decoded_data["skeleton"].split(",")
            posraw=decoded_data["position"].split(",")
            if raw=='untracked' or posraw=='':
                continue
            skeleton=[]
            position=[]
            if raw!='':
                raw=[0]+raw
                for index in read_index:
                    skeleton.append(float(raw[index*7+1]))
                    skeleton.append(float(raw[index*7+2]))
                    skeleton.append(float(raw[index*7+3]))
                if self.one==0:
                    #print self.classifier.dic
                    self.classifier.dic[path].headpos=[int(raw[4]),int(raw[5])]
                    self.classifier.dic[path].shoulder=abs(int(raw[11])-int(raw[25]))
                    self.classifier.dic[path].hip=int(raw[89])
                    self.classifier.dic[path].tall=int(raw[89])-int(raw[5])
                    self.one=1
                if len(position)>5:
                    for i in range(4,6):
                        position.append(int(posraw[i]))
                    for i in range(4,6):
                        position.append(int(posraw[i]))
                else:
                    for i in range(4):
                        position.append(int(posraw[i]))
                    self.classifier.dic[path].positions.append(position)

                self.classifier.dic[path].skeletons.append(skeleton)

            else:

                if len(position)>5:
                    for i in range(4,6):
                        position.append(int(posraw[i]))
                    for i in range(4,6):
                        position.append(int(posraw[i]))
                else:
                    for i in range(4):
                        position.append(int(posraw[i]))
                self.classifier.dic[path].positions.append(position)



            rightimg=None
            leftimg=None
            ftype=decoded_data['label']

            if decoded_data['right']!=None:
                img=decoded_data['right']
                rightimg=img
                cv2.imwrite("/home/lzz/sing/svm/handshaperight/"+str(self.fr)+str(ftype)+".bmp",rightimg)
            if decoded_data['left']!=None:
                img=decoded_data['left']
                leftimg=img
                cv2.imwrite("/home/lzz/sign/svm/handshape/"+str(self.fr)+"l.bmp",leftimg)

            self.classifier.dic[path].dict[self.fr]=frame(self.fr,skeleton,ftype,rightimg,leftimg,position,0)
            self.classifier.dic[path].framelist.append(self.fr)

            self.fr+=1






        if self.classifier.dic[path].skeletons==[] or self.classifier.dic[path].positions==[]:
            self.fr=0
            self.one=0

            self.classifier.reset()
            return

        #self.classifier.dic[path].heightenough()
        self.classifier.dic[path].getBothSeparate()
        self.classifier.dic[path].getInter()
        self.classifier.dic[path].getVelo()

        self.classifier.dic[path].getVelocity()
        self.classifier.dic[path].findTopHandshape()
        self.classifier.getHogFeature()
        try:
            ret=self.classifier.dic[path].consTrajectory()
        except:
            print "Exception in user code:"
            print '-'*60
            traceback.print_exc(file=sys.stdout)
            print '-'*60
            ret=0

        if ret==0:
            self.ret()
            return
        #self.slword.displacement=hodmodule.hod(self.slword.skeletons)
        #self.slword.checkDecisionTreeInter()
        #self.slword.getHogFeature()


        '''import glob
        imgset=glob.glob("/home/lzz/1/*.bmp")

        for i in imgset:
            img=cv2.imread(i)
            img=img/255.0
            self.slword.batch.append(img)'''

        '''print "batch"
        print len(self.slword.batch)
        if len(self.slword.batch)==0:
            self.fr=0
            del self.slword
            self.slword=slWord.SlWord()
            return
        self.caffedl.net.predict(self.slword.batch,False)
        feature=[]
        for s in range(self.slword.keyNo):
            feat = self.caffedl.net.blobs['ip1'].data[s].flatten().tolist()
            feature.append(feat)

        #print feature
        self.slword.handshape=self.slword.pooling(feature,1)'''
        self.classifier.getCaffeFeature(RecProcessor.caffedl.net,RecProcessor.caffedlInter.net)

        self.classifier.separateCaffeFeature()


        '''if self.slword.bothseparate==1:
            leftfeature=[]
            for s in range(self.slword.leftkeyNo):
                feat = self.caffedl.net.blobs['ip1'].data[self.slword.keyNo+s].flatten().tolist()
                leftfeature.append(feat)
            self.slword.lefthandshape=self.slword.pooling(leftfeature,1)'''

        self.classifier.dic[path].combineFeature()
        if self.classifier.dic[path].intersect==0 and self.classifier.dic[path].bothseparate==0:
            print "class 1"
            svmModel= svm_load_model("/home/lzz/sign/svm/single.model")
        elif self.classifier.dic[path].intersect==0 and self.classifier.dic[path].bothseparate==1:
            print "class 2"
            svmModel= svm_load_model("/home/lzz/sign/svm/both.model")
        elif self.classifier.dic[path].intersect==1 and self.classifier.dic[path].bothseparate==0:
            print "class 3"
            svmModel= svm_load_model("/home/lzz/sign/svm/inter.model")
        svm_res1=test_svm_model(svmModel,[0],[self.classifier.dic[path].combinedFeature])
        #print "result",svm_res1
        pred_labels=svm_res1[0]


        print self.signDic[int(pred_labels[0])]
        self.send_data(self.signDic[int(pred_labels[0])]+'#TERMINATOR#')
        self.ret()



    def flatten(self, d, parent_key='', sep='.'):
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, collections.MutableMapping):
                items.extend(self.flatten(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

