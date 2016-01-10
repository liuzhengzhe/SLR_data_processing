from svmutil import *
from svmmodule import *
import random
import time
import binascii
import bz2
import datetime
from FrameConverter import FrameConverter
from echo_server import EchoServer
import collections
import pylab as plt
import cv2
#from caffeDL import *
from Education import *
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
class Edu_Server(object):

    def __init__(self, port):
        f=open('/media/lzz/65c50da0-a3a2-4117-8a72-7b37fd81b574/sign/project/src/EducationNew.pickle')
        self.pickle=pickle.load(f)
        f.close()
        self.fr=0
        self.one=0
        self.exist=0
        self.lock = threading.Lock()
        self.condition=threading.Condition()
        self.cond=threading.Condition()
        self.classifier = EduClassifier()
        self.p_thres=0.4
        self.h_thres=0.5
        self.h_thres_test=0.3


        self.converter = FrameConverter()
        self.caffedl=caffeDL('../../proto/lenet_test_edu.prototxt','../../model/lenet_iter_5000.caffemodel')
        self.caffedlInter=caffeDL('../../proto_inter/lenet_test_edu.prototxt','../../model/lenet__iter_400.caffemodel')
        self.signDic={}
        self.queue = Queue.Queue(maxsize = 10000)

        self.server = EchoServer(port,  self.received_data)
        self.mode=''
        thread.start_new_thread(self.threadfunc,())
        asyncore.loop()

    def received_data(self, received_data):
        decoded_data = self.converter.decode(received_data)
        return self.process_data(decoded_data)




    def process_data(self, decoded_data):

        #print decoded_data
        if decoded_data["label"]=='guide':
            word_name=decoded_data['wordname']
            self.mode='guide'
            self.standard=self.pickle[word_name]
            self.stepno=0

            if self.lock.acquire():
                self.queue.queue.clear()
                self.lock.release()
            self.fr=0
            self.one=0
            self.cond.acquire()
            self.cond.notify()
            self.cond.release()
            self.classifier.reset()
            return '0'
        elif decoded_data['label']=='evaluation':
            print 'evaluation'
            word_name=decoded_data['wordname']
            self.mode='evaluation'
            self.standard=self.pickle[word_name]
            self.queue.queue.clear()
            self.buffer=[]
            self.fr=0
            self.one=0
            self.cond.acquire()
            self.cond.notify()
            self.cond.release()
            print word_name
            self.classifier.reset()
            return '0'
        elif self.mode=='guide':
            self.queue.put(decoded_data)
            self.condition.acquire()
            self.condition.notify()
            self.condition.release()
            #self.process_oneframe(decoded_data)
            return '0'
        elif self.mode=='evaluation':
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
                self.queue.put(decoded_data)
                self.condition.acquire()
                self.condition.notify()
                self.condition.release()
            else:
                self.server.handler.send("Redo"+'#TERMINATOR#')
                if self.lock.acquire():
                    self.queue.queue.clear()
                    self.classifier.buffer=[]
                    self.exist=0
                    self.lock.release()
            return '0'

    def threadfunc(self):
        while(1):

            if(self.mode==''):
                self.cond.acquire()
                self.cond.wait()
                self.cond.release()
            if self.mode=='guide':
                self.condition.acquire()
                if(self.queue.empty()==1):
                    self.condition.wait()
                    self.condition.release()
                while self.queue._qsize()>1:
                    self.queue.get()
                self.process_oneframe(self.queue.get())

            if self.mode=='evaluation':
                self.condition.acquire()
                if(self.queue.empty()==1):
                    self.condition.wait()
                    self.condition.release()
                decoded_data=self.queue.get()
                if self.lock.acquire():
                    self.classifier.buffer.append(decoded_data)
                    self.lock.release()
                if(decoded_data["label"]=="End"):
                    if self.lock.acquire():
                        self.exist=0
                        self.lock.release()
                    if len(self.classifier.buffer)<10:
                        self.server.handler.send("Redo"+"#TERMINATOR#")
                    else:
                        self.process_all()
                    if self.lock.acquire():
                        self.classifier.buffer=[]
                        self.lock.release()


    def process_oneframe(self,decoded_data):
        path="/home/lzz/sign/svm/"
        read_index=[1,3,4,5,6,7,8,9]
        raw=decoded_data["skeleton"].split(",")
        posraw=decoded_data["position"].split(",")
        if raw=='untracked' or posraw=='':
            return
        skeleton=[]
        if raw!='':
            raw=[0]+raw
            for index in read_index:
                skeleton.append(float(raw[index*7+1]))
                skeleton.append(float(raw[index*7+2]))
                skeleton.append(float(raw[index*7+3]))
            if self.one==0:
                self.classifier.dic[path].headpos=[int(raw[4]),int(raw[5])]
                self.classifier.dic[path].shoulder=abs(int(raw[11])-int(raw[25]))
                self.classifier.dic[path].hip=int(raw[89])
                self.classifier.dic[path].tall=int(raw[89])-int(raw[5])
                self.one=1
        if self.stepno>=len(self.standard):
            #print self.stepno,len(self.standard)
            return '0'
        step=self.standard[self.stepno]
        pos0=step['pos']

        pos1=[(float(posraw[0])-self.classifier.dic[path].headpos[0])/self.classifier.dic[path].shoulder,
              (float(posraw[1])-self.classifier.dic[path].headpos[1])/self.classifier.dic[path].tall,
              (float(posraw[2])-self.classifier.dic[path].headpos[0])/self.classifier.dic[path].shoulder,
              (float(posraw[3])-self.classifier.dic[path].headpos[1])/self.classifier.dic[path].tall,
              ]
        tickle=0
        if decoded_data['label']==step['type']:
            if step['type']=='Right':
                #print pos0,pos1

                print 'pos',np.sqrt((pos1[0]-pos0[0])**2+(pos1[1]-pos0[1])**2)
                if np.sqrt((pos1[0]-pos0[0])**2+(pos1[1]-pos0[1])**2)<0.4:
                    start=time.time()
                    h0=step['handshape']
                    img=decoded_data['right']
                    if img!=None:
                        sp=img.shape
                        if sp[0]>sp[1]:
                            img2=cv2.copyMakeBorder(img, 0,0, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2), cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
                        else:
                            img2=cv2.copyMakeBorder(img, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2),0,0, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
                        img3=cv2.resize(img2,(128,128))
                        img3=img3/255.0
                        self.caffedl.net.predict([img3],False)
                        h1 = self.caffedl.net.blobs['ip1'].data[0].flatten().tolist()
                        print 'hand',1 - spatial.distance.cosine(h0,h1)
                        if 1 - spatial.distance.cosine(h0,h1)>self.h_thres:
                            self.stepno+=1
                            tickle=1
                        print 'time:',time.time()-start
            elif step['type']=='Both':
                if np.sqrt((pos1[0]-pos0[0])**2+(pos1[1]-pos0[1])**2)<self.p_thres and np.sqrt((pos1[2]-pos0[2])**2+(pos1[3]-pos0[3])**2)<self.p_thres:
                    h0=step['handshape']
                    lefth0=step['lefthandshape']
                    img=decoded_data['right']
                    leftimg=decoded_data['left']
                    if img!=None and leftimg!=None:
                        sp=img.shape
                        if sp[0]>sp[1]:
                            img2=cv2.copyMakeBorder(img, 0,0, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2), cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
                        else:
                            img2=cv2.copyMakeBorder(img, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2),0,0, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
                        img3=cv2.resize(img2,(128,128))
                        imgr=img3/255.0


                        sp=leftimg.shape
                        if sp[0]>sp[1]:
                            img2=cv2.copyMakeBorder(leftimg, 0,0, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2), cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
                        else:
                            img2=cv2.copyMakeBorder(leftimg, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2),0,0, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
                        img3=cv2.resize(img2,(128,128))
                        imgl=img3/255.0



                        self.caffedl.net.predict([imgr,imgl],False)
                        h1r = self.caffedl.net.blobs['ip1'].data[0].flatten().tolist()
                        h1l = self.caffedl.net.blobs['ip1'].data[1].flatten().tolist()
                        if 1 - spatial.distance.cosine(h0,h1r)>self.h_thres and 1 - spatial.distance.cosine(lefth0,h1l)>self.h_thres:
                            self.stepno+=1
                            tickle=1


            elif step['type']=='Intersect':
                if np.sqrt((pos1[0]-pos0[0])**2+(pos1[1]-pos0[1])**2)<self.p_thres:
                    h0=step['handshape']
                    img=decoded_data['right']
                    if img!=None:
                        sp=img.shape
                        if sp[0]>sp[1]:
                            img2=cv2.copyMakeBorder(img, 0,0, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2), cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
                        else:
                            img2=cv2.copyMakeBorder(img, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2),0,0, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
                        img3=cv2.resize(img2,(128,128))
                        img3=img3/255.0
                        self.caffedlInter.net.predict([img3],False)
                        h1 = self.caffedlInter.net.blobs['ip1'].data[0].flatten().tolist()
                        if 1 - spatial.distance.cosine(h0,h1)>self.h_thres:
                            self.stepno+=1
                            tickle=1
            if tickle==1:
                if self.stepno>=len(self.standard)-1:
                    self.server.handler.send("Finish"+'#TERMINATOR#')
                else:
                    self.server.handler.send("Next"+'#TERMINATOR#')


    def ret(self):
        self.fr=0
        self.one=0

        self.classifier.reset()


    def process_all(self):

        print 'process all'
        buf=self.classifier.buffer
        buffer=buf
        current=0
        ticklelist={}
        cnt=0

        for s in range(len(self.standard)):
            ticklelist[s]={}
        validno=[]
        belong=[]
        for i in range(len(buf)):
            belong.append(-1)

        for s in range(len(self.standard)):
            ticklelist[s]['position']=0
            ticklelist[s]['handshape']=0
            if self.standard[s]['type']=='Both':
                ticklelist[s]['lefthandshape']=0
            for i in range(current,len(buf)-1):
                path="/home/lzz/sign/svm/"
                read_index=[1,3,4,5,6,7,8,9]
                raw=buf[i]["skeleton"].split(",")
                posraw=buf[i]["position"].split(",")
                if raw=='untracked' or posraw=='':
                    return
                skeleton=[]
                if raw!='':
                    raw=[0]+raw
                    for index in read_index:
                        skeleton.append(float(raw[index*7+1]))
                        skeleton.append(float(raw[index*7+2]))
                        skeleton.append(float(raw[index*7+3]))
                    if self.one==0:
                        self.classifier.dic[path].headpos=[int(raw[4]),int(raw[5])]
                        self.classifier.dic[path].shoulder=abs(int(raw[11])-int(raw[25]))
                        self.classifier.dic[path].hip=int(raw[89])
                        self.classifier.dic[path].tall=int(raw[89])-int(raw[5])
                        self.one=1
                if s>=len(self.standard):
                    print s,len(self.standard)
                    return '0'
                pos1=[(float(posraw[0])-self.classifier.dic[path].headpos[0])/self.classifier.dic[path].shoulder,
                      (float(posraw[1])-self.classifier.dic[path].headpos[1])/self.classifier.dic[path].tall,
                      (float(posraw[2])-self.classifier.dic[path].headpos[0])/self.classifier.dic[path].shoulder,
                      (float(posraw[3])-self.classifier.dic[path].headpos[1])/self.classifier.dic[path].tall,
                      ]
                decoded_data=buf[i]
                step=self.standard[s]
                pos0=step['pos']

                tickle=0

                if decoded_data['label']==step['type']:
                    if step['type']=='Right':
                        print 'pos',np.sqrt((pos1[0]-pos0[0])**2+(pos1[1]-pos0[1])**2)
                        if np.sqrt((pos1[0]-pos0[0])**2+(pos1[1]-pos0[1])**2)<self.p_thres:
                            cnt+=1
                            ticklelist[s]['position']=1
                            validno.append(i)
                            belong[i]=s
                            tickle=1


                    elif step['type']=='Both':
                        if np.sqrt((pos1[0]-pos0[0])**2+(pos1[1]-pos0[1])**2)<self.p_thres and np.sqrt((pos1[2]-pos0[2])**2+(pos1[3]-pos0[3])**2)<self.p_thres:
                            ticklelist[s]['position']=1
                            validno.append(i)
                            belong[i]=s
                            tickle=1

                    elif step['type']=='Intersect':
                        if np.sqrt((pos1[0]-pos0[0])**2+(pos1[1]-pos0[1])**2)<self.p_thres:
                            ticklelist[s]['position']=1
                            validno.append(i)
                            belong[i]=s
                            tickle=1

                if tickle==1:
                    current=i

                if i==len(buf)-1:
                    break
        for s in range(len(self.standard)):
            if ticklelist[s]['position']==0:
                self.server.handler.send(str(ticklelist)+'#TERMINATOR#')
                return


        totallen=cnt
        jump=max(1,int(totallen/30))
        cnt=0
        havetickled=[]
        print belong

        for u in range(len(validno)):
            i=validno[u]
            cnt+=1
            step=self.standard[belong[u]]
            if belong[u] in havetickled:
                continue
            if cnt%jump!=0:
                continue
            tickle=0
            decoded_data=buffer[i]

            if step['type']=='Right':
                img=buffer[i]['right']
                h0=step['handshape']
                if img!=None:
                    sp=img.shape
                    if sp[0]>sp[1]:
                        img2=cv2.copyMakeBorder(img, 0,0, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2), cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
                    else:
                        img2=cv2.copyMakeBorder(img, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2),0,0, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
                    img3=cv2.resize(img2,(128,128))
                    img3=img3/255.0
                    self.caffedl.net.predict([img3],False)
                    h1 = self.caffedl.net.blobs['ip1'].data[0].flatten().tolist()
                    print 'hand', 1 - spatial.distance.cosine(h0,h1),belong[i]
                    if 1 - spatial.distance.cosine(h0,h1)>self.h_thres_test:
                        tickle=1
                        print 'tickle',belong[i]
                        ticklelist[belong[i]]['handshape']=1

            elif step['type']=='Both':
                righttickle=0
                lefttickle=0
                img=buffer[i]['right']
                h0=step['handshape']
                if img!=None:
                    sp=img.shape
                    if sp[0]>sp[1]:
                        img2=cv2.copyMakeBorder(img, 0,0, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2), cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
                    else:
                        img2=cv2.copyMakeBorder(img, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2),0,0, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
                    img3=cv2.resize(img2,(128,128))
                    img3=img3/255.0
                    self.caffedl.net.predict([img3],False)
                    h1 = self.caffedl.net.blobs['ip1'].data[0].flatten().tolist()
                    if 1 - spatial.distance.cosine(h0,h1)>self.h_thres_test:
                        righttickle=1
                        ticklelist[belong[i]]['handshape']=1

                img=buffer[i]['left']
                h0=step['lefthandshape']

                if img!=None:
                    sp=img.shape
                    if sp[0]>sp[1]:
                        img2=cv2.copyMakeBorder(img, 0,0, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2), cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
                    else:
                        img2=cv2.copyMakeBorder(img, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2),0,0, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
                    img3=cv2.resize(img2,(128,128))
                    img3=img3/255.0
                    self.caffedl.net.predict([img3],False)
                    h1 = self.caffedl.net.blobs['ip1'].data[0].flatten().tolist()
                    if 1 - spatial.distance.cosine(h0,h1)>self.h_thres_test:
                        lefttickle=1
                        ticklelist[belong[i]]['lefthandshape']=1

                if righttickle==1 and lefttickle==1:
                    tickle=1

            elif step['type']=='Intersect':
                img=buffer[i]['right']

                h0=step['handshape']
                if img!=None:
                    sp=img.shape
                    if sp[0]>sp[1]:
                        img2=cv2.copyMakeBorder(img, 0,0, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2), cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
                    else:
                        img2=cv2.copyMakeBorder(img, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2),0,0, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
                    img3=cv2.resize(img2,(128,128))
                    img3=img3/255.0
                    self.caffedl.net.predict([img3],False)
                    h1 = self.caffedl.net.blobs['ip1'].data[0].flatten().tolist()
                    if 1 - spatial.distance.cosine(h0,h1)>self.h_thres_test:
                        ticklelist[belong[i]]['handshape']=1
                        tickle=1

            if tickle==1:
                havetickled.append(belong[i])

            if i==len(buf)-1:
                break
        self.server.handler.send(str(ticklelist)+'#TERMINATOR#')






    def flatten(self, d, parent_key='', sep='.'):
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, collections.MutableMapping):
                items.extend(self.flatten(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
