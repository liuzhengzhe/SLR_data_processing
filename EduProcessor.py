
import time

import collections
import pylab as plt
import cv2

from Education import *


import thread
import threading
import sys, traceback
from Main_Test import *

class EduProcessor():
    def __init__(self,send_data_callback):
        if hasattr(EduProcessor,'pickle')==0:
            f=open('/media/lzz/65c50da0-a3a2-4117-8a72-7b37fd81b574/sign/project/src/EducationwithDiff.json')
            EduProcessor.pickle=json.load(f)
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
        self.mode=''
        thread.start_new_thread(self.threadfunc,())
        if hasattr(EduProcessor,'caffedl')==0:
            EduProcessor.caffedl=caffeDL('/home/lzz/caffe/caffe-master/examples/imagenet/train_val_16_py_edu.prototxt','/home/lzz/caffe/caffe-master/examples/imagenet/model/4096_iter_10000.caffemodel')
        if hasattr(EduProcessor,'caffedlInter')==0:
            EduProcessor.caffedlInter=EduProcessor.caffedl
        self.cnt=0
        self.signDic={}
        self.queue = Queue.Queue(maxsize = 10000)
        self.send_data=send_data_callback

    def process_data(self, decoded_data):
        if decoded_data["label"]=='guide':
            word_name=decoded_data['wordname']
            self.mode='guide'
            self.standard=EduProcessor.pickle[word_name]
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
            self.standard=EduProcessor.pickle[word_name]
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
            self.cnt+=1
            if self.cnt==5:
                self.cnt=0
                self.queue.put(decoded_data)
                self.condition.acquire()
                self.condition.notify()
                self.condition.release()
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
                self.send_data("Redo"+'#TERMINATOR#')
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
                try:
                    self.process_oneframe(self.queue.get())
                except:
                    pass


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
                        self.send_data("Redo"+"#TERMINATOR#")
                    else:
                        try:
                            self.process_all()
                        except:
                            pass
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
        pos0=step['pos'][4:]

        pos1=[(float(posraw[0])-self.classifier.dic[path].headpos[0])/self.classifier.dic[path].shoulder,
              (float(posraw[1])-self.classifier.dic[path].headpos[1])/self.classifier.dic[path].tall,
              (float(posraw[2])-self.classifier.dic[path].headpos[0])/self.classifier.dic[path].shoulder,
              (float(posraw[3])-self.classifier.dic[path].headpos[1])/self.classifier.dic[path].tall,
              ]
        tickle=0
        postickle=0
        if decoded_data['label']==step['type']:
            if step['type']=='Right':
                #print pos0,pos1

                print 'pos',np.sqrt((pos1[0]-pos0[0])**2+(pos1[1]-pos0[1])**2)
                if np.sqrt((pos1[0]-pos0[0])**2+(pos1[1]-pos0[1])**2)<0.4:
                    postickle=1
                    start=time.time()
                    h0=step['handshape']
                    img=decoded_data['right']
                    if img!=None:
                        sp=img.shape
                        if sp[0]>sp[1]:
                            img2=cv2.copyMakeBorder(img, 0,0, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2), cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
                        else:
                            img2=cv2.copyMakeBorder(img, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2),0,0, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
                        img3=cv2.resize(img2,(227,227))
                        img3=img3/255.0
                        EduProcessor.caffedl.net.predict([img3],False)
                        h1 = EduProcessor.caffedl.net.blobs['fc7'].data[0].flatten().tolist()
                        print 'hand',spatial.distance.cosine(h0,h1),step['diff']
                        if spatial.distance.cosine(h0,h1)<step['diff']+0.1:
                            self.stepno+=1
                            tickle=1
                        print 'time:',time.time()-start
            elif step['type']=='Both':
                print 'pos',np.sqrt((pos1[0]-pos0[0])**2+(pos1[1]-pos0[1])**2),np.sqrt((pos1[2]-pos0[2])**2+(pos1[3]-pos0[3])**2)
                if np.sqrt((pos1[0]-pos0[0])**2+(pos1[1]-pos0[1])**2)<self.p_thres+0.1 and np.sqrt((pos1[2]-pos0[2])**2+(pos1[3]-pos0[3])**2)<self.p_thres+0.1:
                    postickle=1
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
                        img3=cv2.resize(img2,(227,227))
                        imgr=img3/255.0


                        sp=leftimg.shape
                        if sp[0]>sp[1]:
                            img2=cv2.copyMakeBorder(leftimg, 0,0, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2), cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
                        else:
                            img2=cv2.copyMakeBorder(leftimg, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2),0,0, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
                        img3=cv2.resize(img2,(227,227))
                        imgl=img3/255.0



                        EduProcessor.caffedl.net.predict([imgr],False)
                        h1r = EduProcessor.caffedl.net.blobs['fc7'].data[0].flatten().tolist()
                        EduProcessor.caffedl.net.predict([imgl],False)
                        h1l = EduProcessor.caffedl.net.blobs['fc7'].data[0].flatten().tolist()
                        print 'righthand',  spatial.distance.cosine(h0,h1r),step['diff']
                        print 'lefthand',  spatial.distance.cosine(h0,h1l),step['diff']
                        if  spatial.distance.cosine(h0,h1r)<step['diff']+0.15 and  spatial.distance.cosine(lefth0,h1l)<>step['diff']+0.15:
                            self.stepno+=1
                            tickle=1


            elif step['type']=='Intersect':
                #print pos1,pos0
                print 'pos',np.sqrt((pos1[0]-pos0[0])**2+(pos1[1]-pos0[1])**2)
                if np.sqrt((pos1[0]-pos0[0])**2+(pos1[1]-pos0[1])**2)<self.p_thres:
                    postickle=0
                    h0=step['handshape']
                    img=decoded_data['right']
                    if img!=None:
                        sp=img.shape
                        if sp[0]>sp[1]:
                            img2=cv2.copyMakeBorder(img, 0,0, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2), cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
                        else:
                            img2=cv2.copyMakeBorder(img, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2),0,0, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
                        img3=cv2.resize(img2,(227,227))
                        cv2.imwrite('/home/lzz/x.jpg',img3)
                        img3=img3/255.0
                        EduProcessor.caffedlInter.net.predict([img3],False)

                        h1 = EduProcessor.caffedlInter.net.blobs['fc7'].data[0].flatten().tolist()
                        print 'origin',h0[:20],h1[:20]
                        print 'hand',  spatial.distance.cosine(h0,h1),step['diff']
                        if spatial.distance.cosine(h0,h1)<step['diff']+0.1:
                            self.stepno+=1
                            tickle=1
            if postickle==1 and tickle==0:
                self.send_data(str({'type':'guide','position':'1','handshape':'0'})+'#TERMINATOR#')
            elif postickle==0 and tickle==0:
                self.send_data(str({'type':'guide','position':'0','handshape':'0'})+'#TERMINATOR#')
            elif tickle==1:
                self.send_data(str({'type':'guide','position':'1','handshape':'1'})+'#TERMINATOR#')


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
        ticklelist['type']='evaluation'
        ticklelist['data']=[]
        cnt=0

        for s in range(len(self.standard)):
            ticklelist['data'].append({})
        validno=[]
        belong=[]
        for i in range(len(buf)):
            belong.append(-1)

        for s in range(len(self.standard)):
            ticklelist['data'][s]['position']=0
            ticklelist['data'][s]['handshape']=0
            #if self.standard[s]['type']=='Both':
            #    ticklelist['data'][s]['lefthandshape']=0
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
                pos0=step['pos'][4:]

                tickle=0

                if decoded_data['label']==step['type']:
                    if step['type']=='Right':
                        print 'pos',np.sqrt((pos1[0]-pos0[0])**2+(pos1[1]-pos0[1])**2)
                        if np.sqrt((pos1[0]-pos0[0])**2+(pos1[1]-pos0[1])**2)<self.p_thres:
                            cnt+=1
                            ticklelist['data'][s]['position']=1
                            validno.append(i)
                            belong[i]=s
                            tickle=1


                    elif step['type']=='Both':
                        if np.sqrt((pos1[0]-pos0[0])**2+(pos1[1]-pos0[1])**2)<self.p_thres+0.2 and np.sqrt((pos1[2]-pos0[2])**2+(pos1[3]-pos0[3])**2)<self.p_thres+0.2:
                            ticklelist['data'][s]['position']=1
                            validno.append(i)
                            belong[i]=s
                            tickle=1

                    elif step['type']=='Intersect':
                        if np.sqrt((pos1[0]-pos0[0])**2+(pos1[1]-pos0[1])**2)<self.p_thres:
                            ticklelist['data'][s]['position']=1
                            validno.append(i)
                            belong[i]=s
                            tickle=1

                if tickle==1:
                    current=i

                if i==len(buf)-1:
                    break
        print belong
        for s in range(len(self.standard)):
            if ticklelist['data'][s]['position']==0:
                self.send_data(str(ticklelist)+'#TERMINATOR#')
                print ticklelist
                return


        totallen=cnt
        jump=max(1,int(totallen/10*len(self.standard)))
        cnt=0
        havetickled=[]

        print totallen,jump
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
                    img3=cv2.resize(img2,(227,227))
                    img3=img3/255.0
                    EduProcessor.caffedl.net.predict([img3],False)
                    h1 = EduProcessor.caffedl.net.blobs['fc7'].data[0].flatten().tolist()
                    print 'hand',  spatial.distance.cosine(h0,h1),step['diff']
                    if spatial.distance.cosine(h0,h1)<step['diff']+0.1:
                        tickle=1
                        print 'tickle',belong[i]
                        ticklelist['data'][belong[i]]['handshape']=1

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
                    img3=cv2.resize(img2,(227,227))
                    img3=img3/255.0
                    EduProcessor.caffedl.net.predict([img3],False)
                    h1 = EduProcessor.caffedl.net.blobs['fc7'].data[0].flatten().tolist()
                    print 'hand',  spatial.distance.cosine(h0,h1),step['diff']
                    if  spatial.distance.cosine(h0,h1)<step['diff']+0.2:
                        righttickle=1
                        ticklelist['data'][belong[i]]['handshape']=1

                img=buffer[i]['left']
                h0=step['lefthandshape']

                if img!=None:
                    sp=img.shape
                    if sp[0]>sp[1]:
                        img2=cv2.copyMakeBorder(img, 0,0, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2), cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
                    else:
                        img2=cv2.copyMakeBorder(img, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2),0,0, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
                    img3=cv2.resize(img2,(227,227))
                    img3=img3/255.0
                    EduProcessor.caffedl.net.predict([img3],False)
                    h1 = EduProcessor.caffedl.net.blobs['fc7'].data[0].flatten().tolist()
                    print 'hand',  spatial.distance.cosine(h0,h1),step['diff']
                    if spatial.distance.cosine(h0,h1)<step['diff']+0.15:
                        lefttickle=1
                        #ticklelist['data'][belong[i]]['lefthandshape']=1

                if righttickle==1:# and lefttickle==1:
                    tickle=1
                    ticklelist['data'][belong[i]]['handshape']=1

            elif step['type']=='Intersect':
                img=buffer[i]['right']

                h0=step['handshape']
                if img!=None:
                    sp=img.shape
                    if sp[0]>sp[1]:
                        img2=cv2.copyMakeBorder(img, 0,0, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2), cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
                    else:
                        img2=cv2.copyMakeBorder(img, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2),0,0, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
                    img3=cv2.resize(img2,(227,227))
                    img3=img3/255.0
                    EduProcessor.caffedl.net.predict([img3],False)
                    h1 = EduProcessor.caffedl.net.blobs['fc7'].data[0].flatten().tolist()
                    print 'hand',  spatial.distance.cosine(h0,h1),step['diff']
                    if spatial.distance.cosine(h0,h1)<step['diff']+0.1:
                        ticklelist['data'][belong[i]]['handshape']=1
                        tickle=1

            if tickle==1:
                havetickled.append(belong[i])

            if i==len(buf)-1:
                break
        print ticklelist
        self.send_data(str(ticklelist)+'#TERMINATOR#')






    def flatten(self, d, parent_key='', sep='.'):
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, collections.MutableMapping):
                items.extend(self.flatten(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
