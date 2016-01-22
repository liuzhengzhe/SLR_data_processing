from Classifier import *
import json

class SignWordEdu(SignWord):
    def getPart(self,net,net2):

        single=[]
        inter=[]
        for f in self.topIndex:
            if self.dict[f].ftype!='Intersect':
                single.append(f)
            else:
                inter.append(f)
        if self.bothseparate==0 and self.intersect==0:
            batch=[]
            index=[]
            for i in single:
                if len(glob.glob(self.path+"/handshape/"+str(i)+"_*_C*.jpg"))==0:
                    continue
                img=cv2.imread(glob.glob(self.path+"/handshape/"+str(i)+"_*_C*.jpg")[0])
                sp=img.shape
                if sp[0]>sp[1]:
                    img2=cv2.copyMakeBorder(img, 0,0, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2), cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
                else:
                    img2=cv2.copyMakeBorder(img, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2),0,0, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
                img3=cv2.resize(img2,(227,227))
                img3=img3/255.0

                batch.append(img3)
                index.append(i)
            net.predict(batch,False)

            for s in range(len(batch)):
                feat = net.blobs['fc7'].data[s].flatten().tolist()
                self.dict[index[s]].standardhand=feat
        elif self.bothseparate==1:
            batch=[]
            index=[]
            for i in single:
                if len(glob.glob(self.path+"/handshape/"+str(i)+"_*_C*.jpg"))==0:
                    continue
                img=cv2.imread(glob.glob(self.path+"/handshape/"+str(i)+"_*_C*.jpg")[0])
                if img!=None:
                    sp=img.shape
                    if sp[0]>sp[1]:
                        img2=cv2.copyMakeBorder(img, 0,0, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2), cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
                    else:
                        img2=cv2.copyMakeBorder(img, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2),0,0, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
                    img3=cv2.resize(img2,(227,227))
                    img3=img3/255.0

                    batch.append(img3)
                    index.append(i)
            lenr=len(batch)
            for i in single:
                if len(glob.glob(self.path+"/handshape/left/"+str(i)+"_*_C*.jpg"))==0:
                    continue
                img=cv2.imread(glob.glob(self.path+"/handshape/left/"+str(i)+"_*_C*.jpg")[0])
                if img!=None:
                    sp=img.shape
                    if sp[0]>sp[1]:
                        img2=cv2.copyMakeBorder(img, 0,0, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2), cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
                    else:
                        img2=cv2.copyMakeBorder(img, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2),0,0, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
                    img3=cv2.resize(img2,(227,227))
                    img3=img3/255.0

                    batch.append(img3)
                    index.append(i)
            lenl=len(batch)-lenr
            net.predict(batch,False)

            for s in range(lenr):
                feat = net.blobs['fc7'].data[s].flatten().tolist()
                self.dict[index[s]].standardhand=feat
            for s in range(lenr,len(batch)):
                feat = net.blobs['fc7'].data[s].flatten().tolist()
                self.dict[index[s]].standardlefthand=feat

        elif self.intersect==1:
            batch=[]
            index=[]
            for i in inter:
                if len(glob.glob(self.path+"/handshape/"+str(i)+"_*_C*.jpg"))==0:
                    continue
                img=cv2.imread(glob.glob(self.path+"/handshape/"+str(i)+"_*_C*.jpg")[0])
                if img!=None:
                    sp=img.shape
                    if sp[0]>sp[1]:
                        img2=cv2.copyMakeBorder(img, 0,0, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2), cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
                    else:
                        img2=cv2.copyMakeBorder(img, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2),0,0, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
                    img3=cv2.resize(img2,(227,227))
                    img3=img3/255.0

                    batch.append(img3)
                    index.append(i)
            lenr=len(batch)

            net2.predict(batch,False)
            for s in range(lenr):
                feat = net2.blobs['fc7'].data[s].flatten().tolist()
                self.dict[index[s]].standardhand=feat

            batch=[]
            index=[]
            for i in single:
                if len(glob.glob(self.path+"/handshape/"+str(i)+"_*_C*.jpg"))==0:
                    continue
                img=cv2.imread(glob.glob(self.path+"/handshape/"+str(i)+"_*_C*.jpg")[0])
                if img!=None:
                    sp=img.shape
                    if sp[0]>sp[1]:
                        img2=cv2.copyMakeBorder(img, 0,0, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2), cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
                    else:
                        img2=cv2.copyMakeBorder(img, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2),0,0, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
                    img3=cv2.resize(img2,(227,227))
                    img3=img3/255.0

                    batch.append(img3)
                    index.append(i)
            lenr=len(batch)
            for i in single:
                if len(glob.glob(self.path+"/handshape/left/"+str(i)+"_*_C*.jpg"))==0:
                    continue
                img=cv2.imread(glob.glob(self.path+"/handshape/left/"+str(i)+"_*_C*.jpg")[0])
                if img!=None:
                    sp=img.shape
                    if sp[0]>sp[1]:
                        img2=cv2.copyMakeBorder(img, 0,0, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2), cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
                    else:
                        img2=cv2.copyMakeBorder(img, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2),0,0, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
                    img3=cv2.resize(img2,(227,227))
                    img3=img3/255.0

                    batch.append(img3)
                    index.append(i)
            lenl=len(batch)-lenr
            if batch!=[]:
                net.predict(batch,False)

                for s in range(lenr):
                    feat = net.blobs['fc7'].data[s].flatten().tolist()
                    self.dict[index[s]].standardhand=feat
                for s in range(lenr,len(batch)):
                    feat = net.blobs['fc7'].data[s].flatten().tolist()
                    self.dict[index[s]].standardlefthand=feat
        self.steps=[self.topIndex[0]]
        lastframe=self.dict[self.topIndex[0]]
        for i in range(1,len(self.topIndex)):
            pos0=lastframe.position
            pos1=self.dict[self.topIndex[i]].position
            h0=lastframe.standardhand
            h1=self.dict[self.topIndex[i]].standardhand
            print self.topIndex[i],np.sqrt((pos1[0]-pos0[0])**2+(pos1[1]-pos0[1])**2),1 - spatial.distance.cosine(h0,h1)
            if np.sqrt((pos1[0]-pos0[0])**2+(pos1[1]-pos0[1])**2)>25 or 1 - spatial.distance.cosine(h0,h1)<0.2:
                lastframe=self.dict[self.topIndex[i]]
                self.steps.append(self.topIndex[i])
        print self.steps
        stepdetail=[]
        fystep=[]
        for f in self.steps:
            headx=self.headpos[0]
            heady=self.headpos[1]
            shoulder=self.shoulder+0.000001
            tall=self.tall
            dic={}
            dic['sample']=self.sampleName
            dic['frame']=f
            dic['type']=self.dict[f].ftype
            dic['pos']=[self.dict[f].position[0],self.dict[f].position[1],self.dict[f].position[2],self.dict[f].position[3],(self.dict[f].position[0]-headx)/shoulder,(self.dict[f].position[1]-heady)/tall,(self.dict[f].position[2]-headx)/shoulder,(self.dict[f].position[3]-heady)/tall]

            dic['handshape']=self.dict[f].standardhand
            dic['lefthandshape']=self.dict[f].standardlefthand

            fydic={}
            fydic['frame']=f
            fydic['type']=self.dict[f].ftype
            fydic['pos']=[self.dict[f].position[0],self.dict[f].position[1],self.dict[f].position[2],self.dict[f].position[3],(self.dict[f].position[0]-headx)/shoulder,(self.dict[f].position[1]-heady)/tall,(self.dict[f].position[2]-headx)/shoulder,(self.dict[f].position[3]-heady)/tall]

            stepdetail.append(dic)
            fystep.append(fydic)
        return stepdetail,fystep
class EduClassifier(Classifier):

    def listFileEdu(self,path,dic):

        files = os.listdir(path)
        for file in files:
            if(os.path.isdir(path+"/"+file)==1 and file[0:4]=="HKG_"):
                flag=self.testSignWord(path+"/"+file)
                if(flag==0):
                    continue

                signer=file.split(' ')[1]
                if dic.has_key(signer)==0:

                    continue
                if self.have.has_key(file.split(' ')[0]):
                    continue
                self.have[file.split(' ')[0]]=0
                self.filelist.append(path+"/"+file)
                self.dic[path+"/"+file]=SignWordEdu(path+"/"+file,0)
                f=self.dic[path+"/"+file].loadData()
            elif(os.path.isdir(path+"/"+file)==1 and file[0:4]!="HKG_"):
                self.listFileEdu(path+"/"+file,dic)
    def getPart(self):
        self.edudic={}
        self.fydic={}
        for path in self.filelist:
            #if self.edudic.has_key(self.dic[path].wordName):
            if self.dic[path].traintest=='train':
                continue
            print path
            stepdetail,fystep=self.dic[path].getPart(caffedl.net,caffedlInter.net)
            self.edudic[self.dic[path].wordName]=stepdetail
            self.fydic[self.dic[path].sampleName]=fystep

        #with open('/home/lzz/Education.txt', 'w') as outfile:
        #    json.dump(data, outfile)
        with open('EducationNew.json', 'wb') as handle:
          json.dump(self.edudic, handle)
        with open('EducationFy.json', 'wb') as handle:
          json.dump(self.fydic, handle)
if __name__ == '__main__':
    #caffedl=caffeDL('../../proto/lenet_test.prototxt','../../model/lenet_iter_5000.caffemodel')
    #caffedlInter=caffeDL('../../proto_inter/lenet_test.prototxt','../../model/lenet__iter_400.caffemodel')
    caffedl=caffeDL('/home/lzz/caffe/caffe-master/examples/imagenet/train_val_16_py.prototxt','/home/lzz/caffe/caffe-master/examples/imagenet/model/4096_iter_10000.caffemodel')
    caffedlInter=caffeDL('/home/lzz/caffe/caffe-master/examples/imagenet/train_val_16_py.prototxt','/home/lzz/caffe/caffe-master/examples/imagenet/model/4096_iter_10000.caffemodel')

    classifier = EduClassifier()

    dataset='our'

    trainname={}
    testname={}


    if dataset=='devisign':
        #pathTotal='/media/lzz/Data1/devisign/'
        #pathTotal='/media/lzz/Data1/own/'
        pathTotal='/home/lzz/sign/data0/'
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
        #pathTotal='../../data/'
        #pathTotal='/media/lzz/HD1/kinecttry/'
        pathTotal='/media/lzz/65c50da0-a3a2-4117-8a72-7b37fd81b574/sign/data/'
        #pathTotal='/media/lzz/Data1/kinect/similardata/'
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
        #dic['Michael']=0
        #dic['Micheal']=0
        classifier.listFileEdu(pathTotal,dic)

    mode=2


    wholeMode=1


    classifier.getBothSeparate()
    if mode==2:
        classifier.getInter()

    classifier.getVelo()
    classifier.checkType()


    classifier.getVelocity()

    classifier.findTopHandshape()
    classifier.getPart()

    #classifier.constructTrajectory()


    classifier.signkey()
