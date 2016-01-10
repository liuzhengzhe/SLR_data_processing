from Classifier import *
class DtwSignWord(SignWord):
    def dtwfeature(self):
        print self.path
        lamda=1
        last=[0,0,0,0]
        self.trajectory=[]
        if self.bothseparate==0 and self.intersect==0:
            img=self.dict[self.topIndex[0]].rightimg
            sp=img.shape
            if sp[0]>sp[1]:
                img2=cv2.copyMakeBorder(img, 0,0, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2), cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
            else:
                img2=cv2.copyMakeBorder(img, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2),0,0, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
            img3=cv2.resize(img2,(128,128))
            image2 = color.rgb2gray(img3)
            fd, hog_image = hog(image2, orientations=5, pixels_per_cell=(90,90),cells_per_block=(2, 2), visualise=True)

            img=self.dict[self.topIndex[-1]].rightimg
            sp=img.shape
            if sp[0]>sp[1]:
                img2=cv2.copyMakeBorder(img, 0,0, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2), cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
            else:
                img2=cv2.copyMakeBorder(img, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2),0,0, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
            img=cv2.resize(img2,(128,128))
            image2 = color.rgb2gray(img)
            fd2, hog_image = hog(image2, orientations=5, pixels_per_cell=(90,90),cells_per_block=(2, 2), visualise=True)

            self.handhog=fd.tolist()+fd2.tolist()
        else:
            img=self.dict[self.topIndex[0]].rightimg
            sp=img.shape

            if sp[0]>sp[1]:
                img2=cv2.copyMakeBorder(img, 0,0, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2), cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
            else:
                img2=cv2.copyMakeBorder(img, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2),0,0, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
            img=cv2.resize(img2,(128,128))
            img = color.rgb2gray(img)


            fd, hog_image = hog(img, orientations=5, pixels_per_cell=(90,90),cells_per_block=(2, 2), visualise=True)




            img=self.dict[self.topIndex[-1]].rightimg
            sp=img.shape
            if sp[0]>sp[1]:
                img2=cv2.copyMakeBorder(img, 0,0, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2), cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
            else:
                img2=cv2.copyMakeBorder(img, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2),0,0, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
            img=cv2.resize(img2,(128,128))

            img = color.rgb2gray(img)

            fd2, hog_image = hog(img, orientations=5, pixels_per_cell=(90,90),cells_per_block=(2, 2), visualise=True)

            fd3=None
            fd4=None
            start=0
            end=len(self.topIndex)-1
            for i in range(len(self.topIndex)):

                img=self.dict[self.topIndex[0]].leftimg
                if img!=None:
                    sp=img.shape
                    if sp[0]>sp[1]:
                        img2=cv2.copyMakeBorder(img, 0,0, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2), cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
                    else:
                        img2=cv2.copyMakeBorder(img, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2),0,0, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
                    img=cv2.resize(img2,(128,128))

                    img = color.rgb2gray(img)
                    img = color.rgb2gray(img)


                    fd3, hog_image = hog(img, orientations=5, pixels_per_cell=(90,90),cells_per_block=(2, 2), visualise=True)
                    start=i
                    break




            for i in range(len(self.topIndex)-1,0,-1):
                img=self.dict[self.topIndex[-1]].leftimg
                if img!=None:

                    sp=img.shape
                    if sp[0]>sp[1]:
                        img2=cv2.copyMakeBorder(img, 0,0, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2), cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
                    else:
                        img2=cv2.copyMakeBorder(img, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2),0,0, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
                    img=cv2.resize(img2,(128,128))
                    img = color.rgb2gray(img)

                    image2 = color.rgb2gray(img)
                    fd4, hog_image = hog(image2, orientations=5, pixels_per_cell=(90,90),cells_per_block=(2, 2), visualise=True)
                    end=i
                    break
            if fd3==None:
                fd3=fd
            if fd4==None:
                fd4=fd2



            self.handhog=fd.tolist()+fd2.tolist()+fd3.tolist()+fd4.tolist()
        for f in self.framelist[1:-2]:
            if self.bothseparate==0 and self.intersect==0:
                if self.dict.has_key(f-1) and self.dict.has_key(f+1) and self.dict.has_key(f):

                    rightx=(self.dict[f].position[0]-self.headpos[0])/self.shoulder
                    righty=(self.dict[f].position[1]-self.headpos[1])/self.hip-self.headpos[0]


                    dist=math.sqrt(math.pow(self.dict[f+1].position[0]-self.dict[f-1].position[0],2)+math.pow(self.dict[f+1].position[1]-self.dict[f-1].position[1],2))
                    if dist!=0:
                        vx=(self.dict[f+1].position[0]-self.dict[f-1].position[0])/dist
                        vy=(self.dict[f+1].position[1]-self.dict[f-1].position[1])/dist
                    else:
                        vx=0
                        vy=0
                    self.trajedis=[rightx,righty,vx,vy]
                    #last=self.trajedis
                #else:
                    #self.trajedis=last
                    self.trajectory.append(self.trajedis)






            else:
                if self.dict.has_key(f-1) and self.dict.has_key(f+1) and self.dict.has_key(f):
                    rightx=(self.dict[f].position[0]-self.headpos[0])/self.shoulder
                    righty=(self.dict[f].position[1]-self.headpos[1])/(self.hip-self.headpos[0])
                    leftx=(self.dict[f].position[2]-self.headpos[0])/self.shoulder
                    lefty=(self.dict[f].position[3]-self.headpos[1])/(self.hip-self.headpos[0])

                    xdiff=rightx-leftx
                    ydiff=righty-lefty

                    distr=math.sqrt(math.pow(self.dict[f+1].position[0]-self.dict[f-1].position[0],2)+math.pow(self.dict[f+1].position[1]-self.dict[f-1].position[1],2))
                    distl=math.sqrt(math.pow(self.dict[f+1].position[2]-self.dict[f-1].position[2],2)+math.pow(self.dict[f+1].position[3]-self.dict[f-1].position[3],2))
                    if distr==0:
                        vxr=0
                        vyr=0
                    else:
                        vxr=(self.dict[f+1].position[0]-self.dict[f-1].position[0])/distr
                        vyr=(self.dict[f+1].position[1]-self.dict[f-1].position[1])/distr
                    if distl==0:
                        vxl=0
                        vyl=0
                    else:
                        vxl=(self.dict[f+1].position[2]-self.dict[f-1].position[2])/distl
                        vyl=(self.dict[f+1].position[3]-self.dict[f-1].position[3])/distl

                    self.trajedis=[rightx,righty,leftx,lefty,xdiff,ydiff,vxr,vyr,vxl,vyl,vxr-vxl,vyr-vyl]

                    #last=self.trajedis
                #else:
                    #self.trajedis=last
                    self.trajectory.append(self.trajedis)


            #self.combinedFeature=self.trajedis+self.handhog
class DtwClassifier(Classifier):
    def dtwrecog(self):
        trajedic={}
        handdic={}

        trajedicboth={}
        handdicboth={}

        for path in self.trainSet:
            print path
            if self.dic[path].bothseparate==0 and self.dic[path].intersect==0:
                trajedic[path]=self.dic[path].trajectory
                handdic[path]=self.dic[path].handhog

            else:
                trajedicboth[path]=self.dic[path].trajectory
                handdicboth[path]=self.dic[path].handhog
        print 'test'

        lamda=0.3
        correct=0
        wrong=0
        for path in self.testSet:
            print path
            if self.dic[path].bothseparate==0 and self.dic[path].intersect==0:
                dist=float("inf")
                for p in trajedic:
                    dist1=dtw(self.dic[path].trajectory,trajedic[p])[0]
                    dist2=0
                    for i in range(len(self.dic[path].handhog)):
                        dist2+=math.pow(handdic[p][i]-self.dic[path].handhog[i],2)
                    if dist1+dist2*lamda<dist:
                        dist=dist1+dist2*lamda
                        prediction=self.dic[p].wordName
                if self.dic[path].wordName==prediction:
                    correct+=1
                else:
                    wrong+=1
            else:
                dist=float("inf")
                for p in trajedicboth:
                    dist1=dtw(self.dic[path].trajectory,trajedicboth[p])[0]
                    dist2=0
                    for i in range(len(self.dic[path].handhog)):
                        dist2+=math.pow(handdicboth[p][i]-self.dic[path].handhog[i],2)
                    if dist1+dist2*lamda<dist:
                        dist=dist1+dist2*lamda
                        prediction=self.dic[p].wordName
                if self.dic[path].wordName==prediction:
                    correct+=1
                else:
                    wrong+=1
        accuracy=float(correct)/(float(correct+wrong))
        print accuracy
    def dtwfeature(self):
        for path in self.filelist:
            print path
            self.dic[path].dtwfeature()
if __name__ == '__main__':
    classifier = Classifier()
    dataset='our'

    trainname={}
    testname={}

    if dataset=='devisign':
        pathTotal='/media/lzz/Data1/devisign/'
        #pathTotal='/media/lzz/Data1/own/'
        #pathTotal='/home/lzz/sign/data1/'
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
        #pathTotal='/home/lzz/sign/data1/'
        #pathTotal='/media/lzz/Data1/kinect/'
        pathTotal='/media/lzz/HD1/1Michael/split/new/'
        trainname['hfy']=0
        trainname['fuyang']=0

        trainname['Aaron']=0
        trainname['Michael']=0
        trainname['Micheal']=0

        testname['lzz']=0
        dic={}
        dic['Aaron']=0
        dic['Michael']=0
        dic['Micheal']=0
        classifier.listFile(pathTotal)
        classifier.split(trainname,testname,dic,2)

    classifier.constructLabelData()
    classifier.label2Name={}
    for path in classifier.filelist:
        classifier.label2Name[classifier.dic[path].label]=classifier.dic[path].wordName
    classifier.getBothSeparate()
    classifier.getInter()

    classifier.getVelo()
    classifier.getVelocity()

    classifier.findTopHandshape()


    mode=2


    wholeMode=1



    classifier.dtwfeature()
    print 'finish feature'
    classifier.dtwrecog()
