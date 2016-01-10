from caffeDL import *
import os
import scipy
import math
from skimage.feature import hog
from svmmodule import *
import random
import h5py
from SignWord import *
class Classifier():
    def __init__(self):
        self.batch_size=200
        self.namepool={}
        self.index2name={}
        self.label=[]
        self.data=[]
        self.displacement=[]
        self.handshape=[]
        self.hogKey=[]
        self.batch=[]
        self.batch2=[]
        self.filelist=[]
        self.dic={}
        self.buffer=[]
        #data_ind=-1
        self.have={}




    def listFile(self,path):

        files = os.listdir(path)
        for file in files:
            if(os.path.isdir(path+"/"+file)==1 and file[0:4]=="HKG_"):
                flag=self.testSignWord(path+"/"+file)
                if(flag==0):
                    continue
                self.filelist.append(path+"/"+file)



            elif(os.path.isdir(path+"/"+file)==1 and file[0:4]!="HKG_"):
                self.listFile(path+"/"+file)


    def testSignWord(self,path):
        files=os.listdir(path+"/handshape/")


        if(len(files)<5):
            return 0


        #return self.filelist

    def listdevisign(self,path):
        files = os.listdir(path)
        for file in files:
            if(os.path.isdir(path+"/"+file)==1 and file[0]=="P"):
                flag=self.testSignWord(path+"/"+file)
                if(flag==0):
                    continue
                self.filelist.append(path+"/"+file)



            elif(os.path.isdir(path+"/"+file)==1 and file[0]!="P"):
                self.listdevisign(path+"/"+file)


    def addpath(self,path):
        files = os.listdir(path)
        for file in files:
            if(os.path.isdir(path+"/"+file)==1 and file[0:4]=="HKG_"):
                flag=self.testSignWord(path+"/"+file)
                if(flag==0):
                    continue
                self.filelist.append(path+"/"+file)


    def buildDic(self,path,datamode):
        signWord=SignWord(path,datamode)
        self.dic[path]=signWord

    def reset(self):
        try:
            for path in self.filelist:
                del self.dic[path]
        finally:
            self.filelist=[]
            self.filelist.append('/home/lzz/sign/svm/')
            self.dic['/home/lzz/sign/svm/']=SignWord('/home/lzz/sign/svm/',0)
            self.dic['/home/lzz/sign/svm/'].skeletons=[]
            self.dic['/home/lzz/sign/svm/'].positions=[]


    def constructLabelData(self):
        wordNo=-1
        for path in self.filelist:
            word_name=self.dic[path].wordName
            if not self.namepool.has_key(word_name):
                word=wordNo+1
                wordNo+=1
                self.namepool[word_name]=word
                self.index2name[path]=self.dic[path].sampleName
            else:
                word=self.namepool[word_name]
                self.index2name[path]=self.dic[path].sampleName
            self.dic[path].label=word

    def loadfeature(self,w1,w2,w3):
        for path in self.filelist:
            #print path
            self.dic[path].loadfeature(w1,w2,w3)
    def combineFeature(self,w1,w2,w3):
        for path in self.filelist:
            #print path
            self.dic[path].combineFeature(w1,w2,w3)
            #self.label.append(word)
            #self.displacement.append(self.dic[path].combinedFeature)
    def diffInClass(self):
        for path in self.filelist:
            #print path
            f=open(path+'/plt/diff.txt','w')
            for other in self.filelist:
                if self.dic[other].wordName==self.dic[path].wordName and self.dic[other].sampleName!=self.dic[path].sampleName:
                    dif=0
                    disp1=normalize_histogram(self.dic[path].displacement)
                    disp2=normalize_histogram(self.dic[other].displacement)
                    for i in range(len(self.dic[path].displacement)):
                        dif+=(disp1[i]-disp2[i])**2
                    f.write(self.dic[other].sampleName+" "+str(dif)+'\n')
            f.close()

        for path in self.filelist:
            #print path
            f=open(path+'/plt/handshape.txt','w')
            for other in self.filelist:
                if self.dic[other].wordName==self.dic[path].wordName and self.dic[other].sampleName!=self.dic[path].sampleName:
                    dif=0
                    disp1=normalize_histogram(self.dic[path].handshape)
                    disp2=normalize_histogram(self.dic[other].handshape)
                    for i in range(len(self.dic[path].handshape)):
                        dif+=(disp1[i]-disp2[i])**2
                    f.write(self.dic[other].sampleName+" "+str(dif)+'\n')
            f.close()

    def getCaffeFeature(self,net,net2):
        for path in self.filelist:
            for f in self.dic[path].framelist:
                try:
                    del self.dic[path].dict[f].data
                except:
                    pass

        self.featureTotal=[]

        self.featureTotal2=[]






        imgNo=0
        imgNoInter=0
        b=0
        bInter=0
        for path in self.filelist:
                print path
                for i in range(self.dic[path].keyNo):

                    assert len(glob.glob(path+"/handshape/"+str(self.dic[path].topIndex[i])+"_*_C*.jpg"))!=0
                    #continue
                    img=cv2.imread(glob.glob(path+"/handshape/"+str(self.dic[path].topIndex[i])+"_*_C*.jpg")[0])
                    sp=img.shape
                    if sp[0]>sp[1]:
                        img2=cv2.copyMakeBorder(img, 0,0, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2), cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
                    else:
                        img2=cv2.copyMakeBorder(img, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2),0,0, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
                    img3=cv2.resize(img2,(128,128))

                    #cv2.imwrite('/home/lzz/svm/after/'+str(self.dic[path].topIndex[i])+".jpg",img3)
                    #imgs.append(img3)
                    img3=img3/255.0


                    if self.dic[path].dict[self.dic[path].topIndex[i]].ftype!='Intersect':
                        #img3=cv2.resize(img3,(128,128))
                        self.batch.append(img3)
                        imgNo+=1
                    else:
                        self.batch2.append(img3)
                        imgNoInter+=1
                    if(imgNo%self.batch_size==0 and len(self.batch)!=0):
                        print str(b)+"start processing..."
                        net.predict(self.batch,False)
                        print str(b)+"finished processing"
                        for s in range(len(self.batch)):
                            feat = net.blobs['ip1'].data[s].flatten().tolist()

                            self.featureTotal.append(feat)
                        del self.batch
                        self.batch=[]
                        b+=1

                    if(imgNoInter%self.batch_size==0 and len(self.batch2)!=0):
                        print "inter"+str(bInter)+"start processing..."
                        net2.predict(self.batch2,False)
                        print "inter"+str(bInter)+"finished processing"
                        for s in range(len(self.batch2)):
                            feat = net2.blobs['ip1'].data[s].flatten().tolist()
                            self.featureTotal2.append(feat)
                        self.batch2=[]
                        bInter+=1



                for i in range(self.dic[path].leftkeyNo):
                    assert len(glob.glob(path+"/handshape/left/"+str(self.dic[path].lefttopIndex[i])+"_*_C*.jpg"))!=0
                    #continue
                    img=cv2.imread(glob.glob(path+"/handshape/left/"+str(self.dic[path].lefttopIndex[i])+"_*_C*.jpg")[0])
                    #img=cv2.imread(imgname[0])
                    sp=img.shape
                    img2=cv2.copyMakeBorder(img, 0,0, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2), cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
                    img3=cv2.resize(img2,(128,128))
                    img3=img3/255.0
                    #imgs.append(img3)

                    imgNo+=1

                    self.batch.append(img3)
                    if(imgNo%self.batch_size==0):
                        print str(b)+"start processing..."
                        net.predict(self.batch,False)
                        print str(b)+"finished processing"
                        for s in range(len(self.batch)):
                            feat = net.blobs['ip1'].data[s].flatten().tolist()
                            self.featureTotal.append(feat)
                            #feat2= net.blobs['prob'].data[s].flatten().tolist()
                            #self.featureTotal2.append(feat2)
                        del self.batch
                        self.batch=[]
                        b+=1




                #for path in self.filelist:
                '''for f in self.dic[path].framelist:
                    try:
                        del self.dic[path].dict[f].rightimg
                    except:
                        pass
                    try:
                        del self.dic[path].dict[f].leftimg
                    except:
                        pass'''
                del self.dic[path].framelist


        #last batch is not full
        if imgNo%self.batch_size!=0:
                print str(b)+"start processing..."
                net.predict(self.batch,False)
                print str(b)+"finished processing"
                for s in range(len(self.batch)):
                    feat = net.blobs['ip1'].data[s].flatten().tolist()
                    #print feat
                    self.featureTotal.append(feat)
                    #feat2= net.blobs['prob'].data[s].flatten().tolist()
                    #self.featureTotal2.append(feat2)
                self.batch=[]
                b+=1


        if imgNoInter%self.batch_size!=0:
                print str(bInter)+"start processing..."
                net2.predict(self.batch2,False)
                print str(bInter)+"finished processing"
                for s in range(len(self.batch2)):
                    feat = net2.blobs['ip1'].data[s].flatten().tolist()
                    #print feat
                    self.featureTotal2.append(feat)
                    #feat2= net.blobs['prob'].data[s].flatten().tolist()
                    #self.featureTotal2.append(feat2)
                self.batch2=[]
                bInter+=1

                '''_unreachable = gc.collect()
                print 'unreachable object num:%d' % _unreachable
                objgraph.show_most_common_types(limit=50)
                print '0'''''


    def separateCaffeFeature(self):
        img_sum=0
        img_sum_inter=0
        for path in self.filelist:
            self.dic[path].idvdCaffeFeature(img_sum,img_sum_inter,self.featureTotal,self.featureTotal2)
            img_sum+=self.dic[path].singlekeyNo
            img_sum+=self.dic[path].leftkeyNo
            img_sum_inter+=self.dic[path].interkeyNo
            '''if self.dic[path].intersect==0:

                self.dic[path].idvdCaffeFeature(img_sum,self.featureTotal)
                img_sum+=(self.dic[path].keyNo+self.dic[path].leftkeyNo)
            else:
                self.dic[path].idvdCaffeFeatureInter(img_sum_inter,self.featureTotal2)
                img_sum_inter+=self.dic[path].keyNo'''
        del self.featureTotal
        del self.featureTotal2



    #test one of the samples of all person
    def split_data(self):
        trainSet=set()
        testSet=set()
        testLabelSet=set()
        self.singlelen=0
        self.bothlen=0
        self.interlen=0

        for path in self.filelist:
            if ((self.dic[path].label in testLabelSet)==0):

                testSet.add(path)
                self.dic[path].traintest="test"
                testLabelSet.add(self.dic[path].label)

                if self.dic[path].bothseparate==0 and self.dic[path].intersect==0:
                    self.singlelen+=1
                elif self.dic[path].bothseparate==1:
                    self.bothlen+=1
                elif self.dic[path].intersect==1:
                    self.interlen+=1


            else:
                trainSet.add(path)
                self.dic[path].traintest="train"
        return trainSet,testSet


    #test one person all samples
    def splitDataAllOfSomePerson(self,trainname,testname):
        trainSet=set()
        testSet=set()
        trainWords={}
        testWords={}
        filelistnew=[]

        for path in self.filelist:
            sampleName=path.split('/')[-1]
            wordName=sampleName[0:sampleName.find(" ")]
            signer=sampleName[sampleName.find(" ")+1:sampleName.find(" ",sampleName.find(" ")+1)]
            if (trainname.has_key(signer)):
                trainWords[wordName]=0

        for path in self.filelist:
            sampleName=path.split('/')[-1]
            wordName=sampleName[0:sampleName.find(" ")]
            signer=sampleName[sampleName.find(" ")+1:sampleName.find(" ",sampleName.find(" ")+1)]
            if (testname.has_key(signer) and trainWords.has_key(wordName)):
                self.dic[path]=SignWord(path,0)
                f=self.dic[path].loadData()
                if f==1:
                    self.dic[path].traintest="test"
                    filelistnew.append(path)
                    testSet.add(path)
                    testWords[wordName]=0
                else:
                    del self.dic[path]
                #self.dic[path].traintest="test"


        for path in self.filelist:
            sampleName=path.split('/')[-1]
            wordName=sampleName[0:sampleName.find(" ")]
            signer=sampleName[sampleName.find(" ")+1:sampleName.find(" ",sampleName.find(" ")+1)]
            if (trainname.has_key(signer) and testWords.has_key(wordName)):
                self.dic[path]=SignWord(path,0)
                f=self.dic[path].loadData()
                if f==1:
                    self.dic[path].traintest="train"
                    filelistnew.append(path)
                    trainWords[wordName]=0
                    trainSet.add(path)
                else:
                    del self.dic[path]
        print 'lengths',len(trainSet),len(testSet)
        return filelistnew,trainSet,testSet

    #test one person one sample
    def splitDataOneSampleOfAPerson(self,name):
        trainSet=set()
        testSet=set()
        testLabelSet=set()
        filelistnew=[]
        test={}
        for path in self.filelist:
            sampleName=path.split('/')[-1]
            wordName=sampleName[0:sampleName.find(" ")]
            signer=sampleName[sampleName.find(" ")+1:sampleName.find(" ",sampleName.find(" ")+1)]
            if name==signer:
                if (test.has_key(wordName)==0):
                    self.dic[path]=SignWord(path,0)
                    f=self.dic[path].loadData()
                    if f==1:
                        self.dic[path].traintest="test"
                        testLabelSet.add(self.dic[path].label)
                        test[wordName]=0
                        testSet.add(path)
                        filelistnew.append(path)
                    else:
                        del self.dic[path]
                else:

                    self.dic[path]=SignWord(path,0)
                    f=self.dic[path].loadData()
                    if f==1:
                        self.dic[path].traintest="train"
                        filelistnew.append(path)
                        trainSet.add(path)
                    else:
                        del self.dic[path]
        return filelistnew,trainSet,testSet
    def splitxn(self,name):
        trainSet=set()
        testSet=set()
        testLabelSet=set()
        filelistnew=[]
        test={}
        for path in self.filelist:
            sampleName=path.split('/')[-1]
            wordName=sampleName[0:sampleName.find(" ")]
            signer=sampleName[sampleName.find(" ")+1:sampleName.find(" ",sampleName.find(" ")+1)]
            if name==signer:
                if (test.has_key(wordName)==0):
                    self.dic[path]=SignWord(path,0)
                    f=self.dic[path].loadData()
                    if f==1:
                        self.dic[path].traintest="test"
                        testLabelSet.add(self.dic[path].label)
                        test[wordName]=0
                        testSet.add(path)
                        filelistnew.append(path)
                    else:
                        del self.dic[path]
            else:
                self.dic[path]=SignWord(path,0)
                f=self.dic[path].loadData()
                if f==1:
                    self.dic[path].traintest="train"
                    filelistnew.append(path)
                    trainSet.add(path)
                else:
                    del self.dic[path]
        return filelistnew,trainSet,testSet

    def splitDataOneSample(self,name):
        trainSet=set()
        testSet=set()
        testLabelSet=set()
        filelistnew=[]
        test={}
        nosigner=0
        baddata=0
        for path in self.filelist:
            sampleName=path.split('/')[-1]
            wordName=sampleName[0:sampleName.find(" ")]
            signer=sampleName[sampleName.find(" ")+1:sampleName.find(" ",sampleName.find(" ")+1)]
            if name.has_key(signer):
                if (test.has_key(wordName)==0):
                    self.dic[path]=SignWord(path,0)
                    f=self.dic[path].loadData()
                    if f==1:
                        self.dic[path].traintest="test"
                        testLabelSet.add(self.dic[path].label)
                        test[wordName]=0
                        testSet.add(path)
                        filelistnew.append(path)
                    else:

                        del self.dic[path]

                else:

                    self.dic[path]=SignWord(path,0)
                    f=self.dic[path].loadData()
                    if f==1:
                        self.dic[path].traintest="train"
                        filelistnew.append(path)
                        trainSet.add(path)
                    else:


                        del self.dic[path]

            else:
                nosigner+=1
        print len(filelistnew),nosigner,baddata

        return filelistnew,trainSet,testSet
    def split1time(self,name):
        trainSet=set()
        testSet=set()
        testLabelSet=set()
        filelistnew=[]
        test={}
        nosigner=0
        baddata=0
        for path in self.filelist:
            sampleName=path.split('/')[-1]
            wordName=sampleName[0:sampleName.find(" ")]
            signer=sampleName[sampleName.find(" ")+1:sampleName.find(" ",sampleName.find(" ")+1)]
            if name.has_key(signer):
                if (test.has_key(wordName)==0):
                    self.dic[path]=SignWord(path,0)
                    f=self.dic[path].loadData()
                    if f==1:
                        self.dic[path].traintest="train"
                        testLabelSet.add(self.dic[path].label)
                        test[wordName]=0
                        testSet.add(path)
                        filelistnew.append(path)
                    else:

                        del self.dic[path]

                else:

                    self.dic[path]=SignWord(path,0)
                    f=self.dic[path].loadData()
                    if f==1:
                        self.dic[path].traintest="test"
                        filelistnew.append(path)
                        trainSet.add(path)
                    else:


                        del self.dic[path]

            else:
                nosigner+=1
        #print len(filelistnew),nosigner,baddata

        return filelistnew,trainSet,testSet

    def splitDataAllOfSomePersondevisign(self,trainname,testname):
        trainSet=set()
        testSet=set()
        trainWords={}
        testWords={}
        filelistnew=[]

        for path in self.filelist:
            sampleName=path.split('/')[-1]
            wordName=sampleName.split('_')[1]
            signer=sampleName.split('_')[0]
            time=sampleName.split('_')[2]
            if (trainname.has_key(signer) and int(wordName)<1000):
                trainWords[wordName]=0

        for path in self.filelist:
            sampleName=path.split('/')[-1]
            wordName=sampleName.split('_')[1]
            signer=sampleName.split('_')[0]
            time=sampleName.split('_')[2]
            if (testname.has_key(signer) and trainWords.has_key(wordName) and int(wordName)<1000):

                self.dic[path]=SignWord(path,1)
                f=self.dic[path].loadDatadevisign()
                if f==1:
                    self.dic[path].traintest="test"
                    filelistnew.append(path)
                    testWords[wordName]=0
                    testSet.add(path)
                else:
                    del self.dic[path]

        for path in self.filelist:
            print path
            sampleName=path.split('/')[-1]
            wordName=sampleName.split('_')[1]
            signer=sampleName.split('_')[0]
            time=sampleName.split('_')[2]
            if (trainname.has_key(signer) and testWords.has_key(wordName) and int(wordName)<1000):

                self.dic[path]=SignWord(path,1)
                f=self.dic[path].loadDatadevisign()
                if f==1:
                    self.dic[path].traintest="train"
                    filelistnew.append(path)
                    trainWords[wordName]=0
                    trainSet.add(path)
                else:
                    del self.dic[path]
        #print 'traintest:',trainSet,testSet
        return filelistnew,trainSet,testSet


    def splitDataOneSampleOfAPersondevisign(self,name):
        trainSet=set()
        testSet=set()
        testLabelSet=set()
        filelistnew=[]

        for path in self.filelist:
            sampleName=path.split('/')[-1]
            wordName=sampleName.split('_')[1]
            signer=sampleName.split('_')[0]
            if name==signer:
                if (self.dic[path].label in testLabelSet==0):
                    testSet.add(path)
                    self.dic[path]=SignWord(path,1)
                    self.dic[path].traintest="test"
                    testLabelSet.add(self.dic[path].label)
                    filelistnew.append(path)
                else:
                    trainSet.add(path)
                    self.dic[path]=SignWord(path,1)
                    self.dic[path].traintest="train"
                    filelistnew.append(path)
        return filelistnew,trainSet,testSet

    def split(self,trainname,testname,name,mode):
        if mode==0:
            self.filelist,self.trainSet,self.testSet=self.splitDataAllOfSomePerson(trainname,testname)
        elif mode==1:
            self.filelist,self.trainSet,self.testSet=self.splitDataOneSampleOfAPerson(name)
        elif mode==2:
            self.filelist,self.trainSet,self.testSet=self.splitDataOneSample(name)
        elif mode==3:
            self.filelist,self.trainSet,self.testSet=self.splitxn(name)
        elif mode==4:
            self.filelist,self.trainSet,self.testSet=self.split1time(name)

    def splitdevisign(self,trainname,testname,name,mode):
        if mode==0:
            self.filelist,self.trainSet,self.testSet=self.splitDataAllOfSomePersondevisign(trainname,testname)
        else:
            self.filelist,self.trainSet,self.testSet=self.splitDataOneSampleOfAPersondevisign(name)

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






    def saveModel(self):
        '''self.trainSet,self.testSet=self.split_data(intoAccountSet)
        train_labels=[]
        train_data=[]
        test_labels=[]
        test_data=[]
        for path in self.trainSet:
            train_labels.append(self.dic[path].label)
            train_data.append(self.dic[path].combinedFeature)
        for path in self.testSet:
            train_labels.append(self.dic[path].label)
            train_data.append(self.dic[path].combinedFeature)
            test_labels.append(self.dic[path].label)
            test_data.append(self.dic[path].combinedFeature)
        svm_m1=train_svm_model(train_labels,train_data)
        svm_save_model("/home/lzz/svmModel",svm_m1)

        index2Model=open("/home/lzz/ModelIndex.txt","w")
        for path in self.testSet:
            index2Model.write(str(self.dic[path].label)+" "+str(self.dic[path].wordName)+"\n")
        index2Model.close()'''

        train_labels=[]
        train_data=[]
        train_labels_both=[]
        train_data_both=[]
        train_labels_inter=[]
        train_data_inter=[]
        testpathlist=[]
        testpathlist_both=[]
        testpathlist_inter=[]

        for path in self.trainSet:
            if self.dic[path].bothseparate==0 and self.dic[path].intersect==0:
                train_labels.append(self.dic[path].label)
                train_data.append(self.dic[path].combinedFeature)
            elif self.dic[path].bothseparate==1:
                train_labels_both.append(self.dic[path].label)
                train_data_both.append(self.dic[path].combinedFeature)
            elif self.dic[path].intersect==1:
                train_labels_inter.append(self.dic[path].label)
                train_data_inter.append(self.dic[path].combinedFeature)


        for path in self.testSet:
            if self.dic[path].bothseparate==0 and self.dic[path].intersect==0:
                train_labels.append(self.dic[path].label)
                train_data.append(self.dic[path].combinedFeature)
                testpathlist.append(path)
            elif self.dic[path].bothseparate==1:
                train_labels_both.append(self.dic[path].label)
                train_data_both.append(self.dic[path].combinedFeature)
                testpathlist_both.append(path)
            elif self.dic[path].intersect==1:
                train_labels_inter.append(self.dic[path].label)
                train_data_inter.append(self.dic[path].combinedFeature)
                testpathlist_inter.append(path)
        svm_m1=train_svm_model(train_labels,train_data)
        svm_save_model('/home/lzz/sign/svm/single.model',svm_m1)
        del svm_m1
        del train_labels
        del train_data
        svm_m2=train_svm_model(train_labels_both,train_data_both)
        svm_save_model('/home/lzz/sign/svm/both.model',svm_m2)
        del svm_m2
        del train_labels_both
        del train_data_both
        svm_m3=train_svm_model(train_labels_inter,train_data_inter)
        svm_save_model('/home/lzz/sign/svm/inter.model',svm_m3)
        del svm_m3
        del train_labels_inter
        del train_data_inter
        index2Model=open("/home/lzz/sign/svm/ModelIndex.txt","w")
        dictionary=[]
        for path in self.testSet:
            dictionary.append([str(self.dic[path].label),str(self.dic[path].wordName)])
            index2Model.write(str(self.dic[path].label)+" "+str(self.dic[path].wordName)+"\n")
        index2Model.close()


        csvfile = file('/home/lzz/sign/svm/dictionary.csv', 'wb')
        writer = csv.writer(csvfile)
        writer.writerows(dictionary)
        csvfile.close()



    def test_svm(self,w1,w2,w3):
        train_labels_s2=[]
        train_data_s2=[]
        test_labels_s2=[]
        test_data_s2=[]
        train_labels_both=[]
        train_data_both=[]
        test_labels_both=[]
        test_data_both=[]
        train_labels_inter=[]
        train_data_inter=[]
        test_labels_inter=[]
        test_data_inter=[]
        testpathlist=[]
        testpathlist_s2=[]
        testpathlist_both=[]
        testpathlist_inter=[]








        for path in self.trainSet:
            if self.dic[path].bothseparate==0 and self.dic[path].intersect==0:
                train_labels_s2.append(self.dic[path].label)
                train_data_s2.append(self.dic[path].combinedFeature)
            elif self.dic[path].bothseparate==1:
                train_labels_both.append(self.dic[path].label)
                train_data_both.append(self.dic[path].combinedFeature)
            elif self.dic[path].intersect==1:
                train_labels_inter.append(self.dic[path].label)
                train_data_inter.append(self.dic[path].combinedFeature)


        for path in self.testSet:
            if self.dic[path].bothseparate==0 and self.dic[path].intersect==0:
                test_labels_s2.append(self.dic[path].label)
                test_data_s2.append(self.dic[path].combinedFeature)
                testpathlist_s2.append(path)
            elif self.dic[path].bothseparate==1:
                test_labels_both.append(self.dic[path].label)
                test_data_both.append(self.dic[path].combinedFeature)
                testpathlist_both.append(path)
            elif self.dic[path].intersect==1:
                test_labels_inter.append(self.dic[path].label)
                test_data_inter.append(self.dic[path].combinedFeature)
                testpathlist_inter.append(path)


        index2Model=open("/media/lzz/65c50da0-a3a2-4117-8a72-7b37fd81b574/sign/svm/ModelIndex.txt","w")

        dictionary=[]
        for path in self.testSet:
            if [str(self.dic[path].label),str(self.dic[path].wordName)] not in dictionary:
                print str(self.dic[path].wordName)
                dictionary.append([str(self.dic[path].label),str(self.dic[path].wordName)])
                index2Model.write(str(self.dic[path].label)+" "+str(self.dic[path].wordName)+"\n")
        index2Model.close()
        csvfile = file('/media/lzz/65c50da0-a3a2-4117-8a72-7b37fd81b574/sign/svm/dictionary.csv', 'wb')
        writer = csv.writer(csvfile)
        writer.writerows(dictionary)
        csvfile.close()
        wrong=open("/media/lzz/65c50da0-a3a2-4117-8a72-7b37fd81b574/sign/result/wrong.txt","w")
        wrong.write("single:\n")
        for c in range(1000,1001,1):
            wrong.write(str(c)+' ')
            param='-t 0 -c '+str(c)+' -b 1'
            svm_m15=train_svm_model(train_labels_s2,train_data_s2)
            svm_res15=test_svm_model(svm_m15,test_labels_s2,test_data_s2)
            #del train_data_s2

            #del train_labels
            #del train_data
            #del svm_m15


            pred_labels15=svm_res15[0]
            acc15=svm_res15[1][0]
            wrong.write(str(acc15)+':\n')

            for i in range (len(testpathlist_s2)):
                path=testpathlist_s2[i]
                self.dic[path].pred=self.label2Name[int(pred_labels15[i])]



            for i in range (len(testpathlist_s2)):
                path=testpathlist_s2[i]
                if self.dic[path].wordName!=self.dic[path].pred:
                    wrong.write(str(self.dic[path].sampleName)+' '+str(self.dic[path].pred)+'\n')

        for path in self.testSet:
            if self.dic[path].bothseparate==0 and self.dic[path].intersect==0:
                train_labels_s2.append(self.dic[path].label)
                train_data_s2.append(self.dic[path].combinedFeature)
                #testpathlist.append(path)
        svm_m1=train_svm_model(train_labels_s2,train_data_s2)






        svm_save_model('/media/lzz/65c50da0-a3a2-4117-8a72-7b37fd81b574/sign/svm/single.model',svm_m1)
        del svm_m1
        del train_labels_s2
        del train_data_s2

        for path in self.filelist:
            if self.dic[path].bothseparate==0 and self.dic[path].intersect==0:
               del self.dic[path]








        wrong.write("bothseparate:\n")
        for c in range(1000,1001,1):
            wrong.write(str(c)+' ')
            param='-t 0 -c '+str(c)+' -b 1'
            svm_m2=train_svm_model(train_labels_both,train_data_both,param)
            svm_res2=test_svm_model(svm_m2,test_labels_both,test_data_both)
            #del train_data_both
            #del train_labels_both
            #del svm_m2


            pred_labels2=svm_res2[0]
            acc2=svm_res2[1][0]
            wrong.write(str(acc2)+':\n')
            for i in range (len(testpathlist_both)):
                path=testpathlist_both[i]
                self.dic[path].pred=self.label2Name[int(pred_labels2[i])]



            for i in range (len(testpathlist_both)):
                path=testpathlist_both[i]
                if self.dic[path].wordName!=self.dic[path].pred:
                    wrong.write(str(self.dic[path].sampleName)+' '+str(self.dic[path].pred)+'\n')



        for path in self.testSet:
            if self.dic.has_key(path):
                if self.dic[path].bothseparate==1:
                    train_labels_both.append(self.dic[path].label)
                    train_data_both.append(self.dic[path].combinedFeature)
                #testpathlist.append(path)
        svm_m2=train_svm_model(train_labels_both,train_data_both)


        svm_save_model('/media/lzz/65c50da0-a3a2-4117-8a72-7b37fd81b574/sign/svm/both.model',svm_m2)
        del svm_m2
        del train_labels_both
        del train_data_both

        for path in self.filelist:
            if self.dic.has_key(path) and self.dic[path].bothseparate==1:
                del self.dic[path]












        wrong.write("intersect:"+'\n')
        for c in range(1000,1001,1):
            wrong.write(str(c)+' ')
            param='-t 0 -c '+str(c)+' -b 1'

            svm_m3=train_svm_model(train_labels_inter,train_data_inter)
            svm_res3=test_svm_model(svm_m3,test_labels_inter,test_data_inter)
            #del train_data_inter
            #del train_labels_inter
            #del svm_m3
            pred_labels3=svm_res3[0]
            acc3=svm_res3[1][0]
            wrong.write(str(acc3)+':\n')



            for i in range (len(testpathlist_inter)):
                path=testpathlist_inter[i]
                self.dic[path].pred=self.label2Name[int(pred_labels3[i])]


            for i in range (len(testpathlist_inter)):
                path=testpathlist_inter[i]
                if self.dic[path].wordName!=self.dic[path].pred:
                    wrong.write(str(self.dic[path].sampleName)+' '+str(self.dic[path].pred)+'\n')
        for path in self.testSet:
            if self.dic.has_key(path):
                if self.dic[path].intersect==1:
                    train_labels_inter.append(self.dic[path].label)
                    train_data_inter.append(self.dic[path].combinedFeature)

        svm_m3=train_svm_model(train_labels_inter,train_data_inter)

        svm_save_model('/media/lzz/65c50da0-a3a2-4117-8a72-7b37fd81b574/sign/svm/inter.model',svm_m3)
        del svm_m3
        del train_labels_inter
        del train_data_inter
        for path in self.filelist:
            if self.dic.has_key(path) and self.dic[path].intersect==1:
                del self.dic[path]

        result=open("/media/lzz/65c50da0-a3a2-4117-8a72-7b37fd81b574/sign/result/result"+str(w1)+str(w2)+str(w3)+".txt","w")
        print "acc15:",acc15,"len15",len(pred_labels15),"acc2:",acc2,"len2",len(pred_labels2),"acc3:",acc3,"len3",len(pred_labels3)
        totalAccuracy=(acc15*len(pred_labels15)+acc2*len(pred_labels2)+acc3*len(pred_labels3))/(len(pred_labels15)+len(pred_labels2)+len(pred_labels3))
        result.write(str(acc15))
        result.write(str(acc2))
        result.write(str(acc3))
        result.write(str(totalAccuracy))
        result.close()
        print "totalaccuracy:",totalAccuracy



        f1=open("/media/lzz/65c50da0-a3a2-4117-8a72-7b37fd81b574/sign/result/wrong.txt","r")
        f2=open("/media/lzz/65c50da0-a3a2-4117-8a72-7b37fd81b574/sign/result/wrongsort.txt","w")
        data=f1.readlines()
        data.sort()
        for d in data:
            f2.write(d)
    def test_svm_load(self,w1,w2,w3):
        train_labels_s2=[]
        train_data_s2=[]
        test_labels_s2=[]
        test_data_s2=[]
        train_labels_both=[]
        train_data_both=[]
        test_labels_both=[]
        test_data_both=[]
        train_labels_inter=[]
        train_data_inter=[]
        test_labels_inter=[]
        test_data_inter=[]
        testpathlist=[]
        testpathlist_s2=[]
        testpathlist_both=[]
        testpathlist_inter=[]








        '''for path in self.trainSet:
            if self.dic[path].bothseparate==0 and self.dic[path].intersect==0:
                train_labels_s2.append(self.dic[path].label)
                train_data_s2.append(self.dic[path].combinedFeature)
            elif self.dic[path].bothseparate==1:
                train_labels_both.append(self.dic[path].label)
                train_data_both.append(self.dic[path].combinedFeature)
            elif self.dic[path].intersect==1:
                train_labels_inter.append(self.dic[path].label)
                train_data_inter.append(self.dic[path].combinedFeature)'''


        for path in self.testSet:
            if self.dic[path].bothseparate==0 and self.dic[path].intersect==0:
                test_labels_s2.append(self.dic[path].label)
                test_data_s2.append(self.dic[path].combinedFeature)
                testpathlist_s2.append(path)
            elif self.dic[path].bothseparate==1:
                test_labels_both.append(self.dic[path].label)
                test_data_both.append(self.dic[path].combinedFeature)
                testpathlist_both.append(path)
            elif self.dic[path].intersect==1:
                test_labels_inter.append(self.dic[path].label)
                test_data_inter.append(self.dic[path].combinedFeature)
                testpathlist_inter.append(path)


        index2Model=open("/home/lzz/sign/svm/ModelIndex.txt","w")

        dictionary=[]
        for path in self.testSet:
            if [str(self.dic[path].label),str(self.dic[path].wordName)] not in dictionary:
                print str(self.dic[path].wordName)
                dictionary.append([str(self.dic[path].label),str(self.dic[path].wordName)])
                index2Model.write(str(self.dic[path].label)+" "+str(self.dic[path].wordName)+"\n")
        index2Model.close()
        '''csvfile = file('/home/lzz/sign/svm/dictionary.csv', 'wb')
        writer = csv.writer(csvfile)
        writer.writerows(dictionary)
        csvfile.close()'''
        self.signDic={}
        Label1=open('/home/lzz/sign/svm/dictionary.csv','rb')
        reader = csv.reader(Label1)
        for row in reader:
            self.signDic[int(row[0])]=row[1]
        wrong=open("/home/lzz/sign/result/wrong.txt","w")
        wrong.write("single:\n")
        for c in range(1000,1001,1):
            wrong.write(str(c)+' ')
            param='-t 0 -c '+str(c)+' -b 1'
            #svm_m15=train_svm_model(train_labels_s2,train_data_s2)
            svm_m15= svm_load_model("/home/lzz/sign/svm/single.model")
            svm_res15=test_svm_model(svm_m15,test_labels_s2,test_data_s2)
            #del train_data_s2

            #del train_labels
            #del train_data
            #del svm_m15


            pred_labels15=svm_res15[0]
            acc15=svm_res15[1][0]
            wrong.write(str(acc15)+':\n')

            for i in range (len(testpathlist_s2)):
                path=testpathlist_s2[i]
                self.dic[path].pred=self.label2Name[int(pred_labels15[i])]



            for i in range (len(testpathlist_s2)):
                path=testpathlist_s2[i]
                if self.dic[path].wordName!=self.dic[path].pred:
                    wrong.write(str(self.dic[path].sampleName)+' '+str(self.dic[path].pred)+'\n')


        del train_labels_s2
        del train_data_s2

        for path in self.filelist:
            if self.dic[path].bothseparate==0 and self.dic[path].intersect==0:
               del self.dic[path]








        wrong.write("bothseparate:\n")
        for c in range(1000,1001,1):
            wrong.write(str(c)+' ')
            param='-t 0 -c '+str(c)+' -b 1'
            #svm_m2=train_svm_model(train_labels_both,train_data_both,param)
            svm_m2= svm_load_model("/home/lzz/sign/svm/both.model")
            svm_res2=test_svm_model(svm_m2,test_labels_both,test_data_both)
            #del train_data_both
            #del train_labels_both
            #del svm_m2


            pred_labels2=svm_res2[0]
            acc2=svm_res2[1][0]
            wrong.write(str(acc2)+':\n')
            for i in range (len(testpathlist_both)):
                path=testpathlist_both[i]
                self.dic[path].pred=self.label2Name[int(pred_labels2[i])]



            for i in range (len(testpathlist_both)):
                path=testpathlist_both[i]
                if self.dic[path].wordName!=self.dic[path].pred:
                    wrong.write(str(self.dic[path].sampleName)+' '+str(self.dic[path].pred)+'\n')



        for path in self.testSet:
            if self.dic.has_key(path):
                if self.dic[path].bothseparate==1:
                    train_labels_both.append(self.dic[path].label)
                    train_data_both.append(self.dic[path].combinedFeature)
                #testpathlist.append(path)
        svm_m2=train_svm_model(train_labels_both,train_data_both)


        svm_save_model('/home/lzz/sign/svm/both.model',svm_m2)
        del svm_m2
        del train_labels_both
        del train_data_both

        for path in self.filelist:
            if self.dic.has_key(path) and self.dic[path].bothseparate==1:
                del self.dic[path]












        wrong.write("intersect:"+'\n')
        for c in range(1000,1001,1):
            wrong.write(str(c)+' ')
            param='-t 0 -c '+str(c)+' -b 1'

            #svm_m3=train_svm_model(train_labels_inter,train_data_inter)
            svm_m3= svm_load_model("/home/lzz/sign/svm/inter.model")
            svm_res3=test_svm_model(svm_m3,test_labels_inter,test_data_inter)
            #del train_data_inter
            #del train_labels_inter
            #del svm_m3
            pred_labels3=svm_res3[0]
            acc3=svm_res3[1][0]
            wrong.write(str(acc3)+':\n')



            for i in range (len(testpathlist_inter)):
                path=testpathlist_inter[i]
                self.dic[path].pred=self.label2Name[int(pred_labels3[i])]


            for i in range (len(testpathlist_inter)):
                path=testpathlist_inter[i]
                if self.dic[path].wordName!=self.dic[path].pred:
                    wrong.write(str(self.dic[path].sampleName)+' '+str(self.dic[path].pred)+'\n')
        for path in self.testSet:
            if self.dic.has_key(path):
                if self.dic[path].intersect==1:
                    train_labels_inter.append(self.dic[path].label)
                    train_data_inter.append(self.dic[path].combinedFeature)

        svm_m3=train_svm_model(train_labels_inter,train_data_inter)

        svm_save_model('/home/lzz/sign/svm/inter.model',svm_m3)
        del svm_m3
        del train_labels_inter
        del train_data_inter
        for path in self.filelist:
            if self.dic.has_key(path) and self.dic[path].intersect==1:
                del self.dic[path]

        result=open("/home/lzz/sign/result/result"+str(w1)+str(w2)+str(w3)+".txt","w")
        print "acc15:",acc15,"len15",len(pred_labels15),"acc2:",acc2,"len2",len(pred_labels2),"acc3:",acc3,"len3",len(pred_labels3)
        totalAccuracy=(acc15*len(pred_labels15)+acc2*len(pred_labels2)+acc3*len(pred_labels3))/(len(pred_labels15)+len(pred_labels2)+len(pred_labels3))
        result.write(str(acc15))
        result.write(str(acc2))
        result.write(str(acc3))
        result.write(str(totalAccuracy))
        result.close()
        print "totalaccuracy:",totalAccuracy



        f1=open("/home/lzz/sign/result/wrong.txt","r")
        f2=open("/home/lzz/sign/result/wrongsort.txt","w")
        data=f1.readlines()
        data.sort()
        for d in data:
            f2.write(d)







    def checkDecisionTreeInter(self):


        '''for path in self.filelist:
            if path in self.trainSet:
                if self.dic[path].intersect==1:
                    for p in self.filelist:
                        if p in self.trainSet:
                            if self.dic[p].wordName==self.dic[path].wordName and self.dic[p].intersect==0 and self.dic[p].bothseparate==1:
                                self.dic[path].intersect=0
                                self.dic[path].shouldgenerate=0
                                self.dic[path].bothseparate=1
                                self.dic[path].leftkeyNo=10
                            elif self.dic[p].wordName==self.dic[path].wordName and self.dic[p].intersect==0 and self.dic[p].bothseparate==0:
                                self.dic[path].intersect=0
                                self.dic[path].shouldgenerate=0
                                self.dic[path].bothseparate=0
                                self.dic[path].leftkeyNo=0'''

        for path in self.filelist:
            if self.dic[path].intersect==1 and self.dic[path].shouldgenerate==1:
                for f in self.dic[path].alsointerframe:
                    output=self.dic[path].createIntersectImage(f)

        print self.dic[path].bothseparate
        print self.dic[path].intersect

    def checkDecisionTreeBothSingle(self):
        for path in self.filelist:
            if path in self.trainSet:
                if self.dic[path].bothseparate==1:
                    for p in self.filelist:
                        if p in self.trainSet:
                            if self.dic[p].wordName==self.dic[path].wordName and self.dic[p].bothseparate==0 and self.dic[p].intersect==0:
                                self.dic[path].bothseparate=0
                                self.dic[path].leftkeyNo=0



    def getInter(self):
        for path in self.filelist:
            self.dic[path].getInter()
            #print self.dic[path].bothseparate
            #print self.dic[path].intersect
    def getInterdtw(self):
        for path in self.filelist:
            self.dic[path].getInterdtw()
    def getBothSeparate(self):
        for path in self.filelist:
            self.dic[path].getBothSeparate()

    def getVelocity(self):
        for path in self.filelist:
            self.dic[path].getVelocity()
    def getTopPos(self):
        for path in self.filelist:
            self.dic[path].findTopPos()
    def findTopHandshape(self):
        for path in self.filelist:
            self.dic[path].findTopHandshape()

    def checkType(self):
        for path in self.filelist:
            assert (self.dic[path].intersect==1 and self.dic[path].bothseparate==1)==0


    def getHogFeature(self):
        for path in self.filelist:
            self.dic[path].getHogFeature()

    def constructFeature(self):
        for path in self.filelist:
            self.dic[path].consFeature()
    def constructTrajectory(self):
        for path in self.filelist:
            self.dic[path].consTrajectory()
    def heightenough(self):
        for path in self.filelist:
            self.dic[path].heightenough()
    def getVelo(self):
        for path in self.filelist:
            self.dic[path].getVelo()
    def signkey(self):
        for path in self.filelist:
            self.dic[path].modifyKeySign()

    def makeXml(self):

        doc = Document()

        total = doc.createElement('handShapeFeatures')
        #bookstore.setAttribute('xmlns:xsi',"http://www.w3.org/2001/XMLSchema-instance")
        #bookstore.setAttribute('xsi:noNamespaceSchemaLocation','bookstore.xsd')
        doc.appendChild(total)

        for path in self.filelist:
            #print path
            if self.dic[path].traintest=='test':
                word = doc.createElement('word')
                word_name = doc.createElement('name')
                word_text = doc.createTextNode(self.dic[path].wordName)
                word_name.appendChild(word_text)
                word.appendChild(word_name)
                total.appendChild(word)

                type_name = doc.createElement('type')
                if self.dic[path].bothseparate==0 and self.dic[path].intersect==0:
                    type_text = doc.createTextNode('single')
                elif self.dic[path].bothseparate==1:
                    type_text = doc.createTextNode('both')
                elif self.dic[path].intersect==1:
                    type_text = doc.createTextNode('intersect')
                word.appendChild(type_name)
                type_name.appendChild(type_text)


                feature = doc.createElement('feature')
                word.appendChild(feature)
                for i in range(500):
                    author_first_name = doc.createElement('value')
                    author_first_name_text = doc.createTextNode(str(self.dic[path].handshape[i]))
                    feature.appendChild(author_first_name)
                    author_first_name.appendChild(author_first_name_text)
                word.appendChild(feature)
                if self.dic[path].bothseparate==1:
                    feature = doc.createElement('leftfeature')
                    word.appendChild(feature)
                    for i in range(500):
                        author_first_name = doc.createElement('value')
                        author_first_name_text = doc.createTextNode(str(self.dic[path].lefthandshape[i]))
                        feature.appendChild(author_first_name)
                        author_first_name.appendChild(author_first_name_text)
                    word.appendChild(feature)






        f = open('feature_sample.xml','w')
        f.write(doc.toprettyxml(indent = ''))
        f.close()
    def savehdf5(self):
        ftrain=open('/media/lzz/65c50da0-a3a2-4117-8a72-7b37fd81b574/sign/proto_hdf5/hdf5train','w')
        ftest=open('/media/lzz/65c50da0-a3a2-4117-8a72-7b37fd81b574/sign/proto_hdf5/hdf5test','w')
        ftrain.write('/media/lzz/65c50da0-a3a2-4117-8a72-7b37fd81b574/sign/proto_hdf5/hdf5/train.h5')
        ftest.write('/media/lzz/65c50da0-a3a2-4117-8a72-7b37fd81b574/sign/proto_hdf5/hdf5/test.h5')

        datas=[]
        labels=[]
        datavalid=[]
        labelvalid=[]
        datatest=[]
        labeltest=[]





        for path in self.filelist:
            if self.dic[path].traintest=='train':
                if random.random()>0.1:
                    datas.append(self.dic[path].handshape)
                    labels.append(self.dic[path].label)
                else:
                    datavalid.append(self.dic[path].handshape)
                    labelvalid.append(self.dic[path].label)
            else:
                datatest.append(self.dic[path].handshape)
                labeltest.append(self.dic[path].label)
        datas=np.array(datas)*1000.0
        labels=np.array(labels).astype(np.float32).transpose()
        datavalid=np.array(datavalid)*1000.0
        labelvalid=np.array(labelvalid).astype(np.float32).transpose()
        datatest=np.array(datatest)*1000.0
        labeltest=np.array(labeltest).astype(np.float32).transpose()
        with h5py.File('/media/lzz/65c50da0-a3a2-4117-8a72-7b37fd81b574/sign/proto_hdf5/hdf5/train.h5', 'w') as f:
            f['data'] = np.asarray(datas)
            f['label'] = np.asarray(labels)
        with h5py.File('/media/lzz/65c50da0-a3a2-4117-8a72-7b37fd81b574/sign/proto_hdf5/hdf5/valid.h5', 'w') as f:
            f['data'] = np.asarray(datavalid)
            f['label'] = np.asarray(labelvalid)
        with h5py.File('/media/lzz/65c50da0-a3a2-4117-8a72-7b37fd81b574/sign/proto_hdf5/hdf5/test.h5', 'w') as f:
            f['data'] = np.asarray(datatest)
            f['label'] = np.asarray(labeltest)
        fnum=open('./number.txt','w')
        fnum.write(str(np.amax(labels)))

    def trajehdf5(self):

        ftrain=open('/media/lzz/65c50da0-a3a2-4117-8a72-7b37fd81b574/sign/proto_hdf5/trajetrain','w')
        ftest=open('/media/lzz/65c50da0-a3a2-4117-8a72-7b37fd81b574/sign/proto_hdf5/trajetest','w')
        ftrain.write('/media/lzz/65c50da0-a3a2-4117-8a72-7b37fd81b574/sign/proto_hdf5/hdf5/trajetrain.h5')
        ftest.write('/media/lzz/65c50da0-a3a2-4117-8a72-7b37fd81b574/sign/proto_hdf5/hdf5/trajetest.h5')

        datas=[]
        labels=[]
        datavalid=[]
        labelvalid=[]
        datatest=[]
        labeltest=[]
        datas2=[]
        labels2=[]
        datavalid2=[]
        labelvalid2=[]
        datatest2=[]
        labeltest2=[]
        datas3=[]
        labels3=[]
        datavalid3=[]
        labelvalid3=[]
        datatest3=[]
        labeltest3=[]




        for path in self.filelist:
            if self.dic[path].bothseparate==0 and self.dic[path].intersect==0:
                if self.dic[path].traintest=='train':
                    if len(self.dic[path].xtrajectory+self.dic[path].ytrajectory)!=400:
                        print path
                        continue
                    if random.random()>0.1:
                        datas.append(self.dic[path].xtrajectory+self.dic[path].ytrajectory)
                        labels.append(self.dic[path].label)
                    else:
                        datavalid.append(self.dic[path].xtrajectory+self.dic[path].ytrajectory)
                        labelvalid.append(self.dic[path].label)
                else:
                    datatest.append(self.dic[path].xtrajectory+self.dic[path].ytrajectory)
                    labeltest.append(self.dic[path].label)
            elif self.dic[path].bothseparate==1:
                if self.dic[path].traintest=='train':
                    if len(self.dic[path].xtrajectory+self.dic[path].ytrajectory)!=400:
                        print path
                        continue
                    if random.random()>0.1:
                        datas2.append(self.dic[path].xtrajectory+self.dic[path].ytrajectory)
                        labels2.append(self.dic[path].label)
                    else:
                        datavalid2.append(self.dic[path].xtrajectory+self.dic[path].ytrajectory)
                        labelvalid2.append(self.dic[path].label)
                else:
                    datatest2.append(self.dic[path].xtrajectory+self.dic[path].ytrajectory)
                    labeltest2.append(self.dic[path].label)
            elif self.dic[path].intersect==1:
                if self.dic[path].traintest=='train':
                    if len(self.dic[path].xtrajectory+self.dic[path].ytrajectory)!=400:
                        print path
                        continue
                    if random.random()>0.1:
                        datas3.append(self.dic[path].xtrajectory+self.dic[path].ytrajectory)
                        labels3.append(self.dic[path].label)
                    else:
                        datavalid3.append(self.dic[path].xtrajectory+self.dic[path].ytrajectory)
                        labelvalid3.append(self.dic[path].label)
                else:
                    datatest3.append(self.dic[path].xtrajectory+self.dic[path].ytrajectory)
                    labeltest3.append(self.dic[path].label)
        datas=np.array(datas).astype(np.float32)*1000.0
        datas2=np.array(datas2).astype(np.float32)*1000.0
        datas3=np.array(datas3).astype(np.float32)*1000.0
        labels=np.array(labels).astype(np.float32).transpose()
        labels2=np.array(labels2).astype(np.float32).transpose()
        labels3=np.array(labels3).astype(np.float32).transpose()
        datavalid=np.array(datavalid).astype(np.float32)*1000.0
        datavalid2=np.array(datavalid2).astype(np.float32)*1000.0
        datavalid3=np.array(datavalid3).astype(np.float32)*1000.0
        labelvalid=np.array(labelvalid).astype(np.float32).transpose()
        labelvalid2=np.array(labelvalid2).astype(np.float32).transpose()
        labelvalid3=np.array(labelvalid3).astype(np.float32).transpose()
        datatest=np.array(datatest).astype(np.float32)*1000.0
        datatest2=np.array(datatest2).astype(np.float32)*1000.0
        datatest3=np.array(datatest3).astype(np.float32)*1000.0
        labeltest=np.array(labeltest).astype(np.float32).transpose()
        labeltest2=np.array(labeltest2).astype(np.float32).transpose()
        labeltest3=np.array(labeltest3).astype(np.float32).transpose()
        with h5py.File('/media/lzz/65c50da0-a3a2-4117-8a72-7b37fd81b574/sign/proto_hdf5/hdf5/trajetrain.h5', 'w') as f:
            f['data'] = np.asarray(datas)
            f['label'] = np.asarray(labels)
        with h5py.File('/media/lzz/65c50da0-a3a2-4117-8a72-7b37fd81b574/sign/proto_hdf5/hdf5/trajevalid.h5', 'w') as f:
            f['data'] = np.asarray(datavalid)
            f['label'] = np.asarray(labelvalid)
        with h5py.File('/media/lzz/65c50da0-a3a2-4117-8a72-7b37fd81b574/sign/proto_hdf5/hdf5/trajetest.h5', 'w') as f:
            f['data'] = np.asarray(datatest)
            f['label'] = np.asarray(labeltest)


        with h5py.File('/media/lzz/65c50da0-a3a2-4117-8a72-7b37fd81b574/sign/proto_hdf5/hdf5/trajetrainboth.h5', 'w') as f:
            f['data'] = np.asarray(datas2)
            f['label'] = np.asarray(labels2)
        with h5py.File('/media/lzz/65c50da0-a3a2-4117-8a72-7b37fd81b574/sign/proto_hdf5/hdf5/trajevalidboth.h5', 'w') as f:
            f['data'] = np.asarray(datavalid2)
            f['label'] = np.asarray(labelvalid2)
        with h5py.File('/media/lzz/65c50da0-a3a2-4117-8a72-7b37fd81b574/sign/proto_hdf5/hdf5/trajetestboth.h5', 'w') as f:
            f['data'] = np.asarray(datatest2)
            f['label'] = np.asarray(labeltest2)


        with h5py.File('/media/lzz/65c50da0-a3a2-4117-8a72-7b37fd81b574/sign/proto_hdf5/hdf5/trajetraininter.h5', 'w') as f:
            f['data'] = np.asarray(datas3)
            f['label'] = np.asarray(labels3)
        with h5py.File('/media/lzz/65c50da0-a3a2-4117-8a72-7b37fd81b574/sign/proto_hdf5/hdf5/trajevalidinter.h5', 'w') as f:
            f['data'] = np.asarray(datavalid3)
            f['label'] = np.asarray(labelvalid3)
        with h5py.File('/media/lzz/65c50da0-a3a2-4117-8a72-7b37fd81b574/sign/proto_hdf5/hdf5/trajetestinter.h5', 'w') as f:
            f['data'] = np.asarray(datatest3)
            f['label'] = np.asarray(labeltest3)
        fnum=open('./number.txt','w')
        fnum.write(str(np.amax(labels)))





    def getDifficulty(self):
        diff={}
        num={}
        f=open('/media/lzz/65c50da0-a3a2-4117-8a72-7b37fd81b574/sign/project/src/EducationNew.pickle')
        edudic=pickle.load(f)
        for path in self.filelist:
            if self.dic[path].traintest=='train':
                difficult=self.dic[path].getdiffi(edudic)
                if diff.has_key(path)==0:
                    diff[path] = []
                    for i in range(len(difficult)):
                        diff[path].append(0)
                    num[path]=0
                for i in range(len(difficult)):
                    diff[path][i]+=difficult[i]
                num[path]+=1

        for k in diff.keys():
            for i in range(len(diff[k])):
                diff[k][i]/=float(num[k])
        with open('difficulty.pickle', 'wb') as handle:
          pickle.dump(diff, handle)


    def enlarge(self,f):
        for path in self.filelist:
            self.dic[path].enlarge(f)