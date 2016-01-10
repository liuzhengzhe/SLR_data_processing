from sklearn.decomposition import PCA
from Classifier import *
#Yin Fang

class SignWordYin(SignWord):
    def buildpca(self,frames):

        hogSet=[]
        print self.path


        for f in frames:
            frame=self.dict[f]
            image=frame.rightimg
            if image!=None:
                sp=image.shape
                if sp[0]>sp[1]:
                    img2=cv2.copyMakeBorder(image, 0,0, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2), cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
                else:
                    img2=cv2.copyMakeBorder(image, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2),0,0, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
                img3=cv2.resize(img2,(128,128))
                image=img3
                image = color.rgb2gray(image)
                fd, hog_image = hog(image, orientations=9, pixels_per_cell=(32,32),cells_per_block=(2, 2), visualise=True)

                hogSet.append(fd)
        if hogSet==[]:
            hogSet=self.hogset[-1]

        self.hogset.append(hogSet)
        return hogSet







    def getHogYin(self,hogSet):

        '''hogSet=[]
        print self.path


        for f in frames:
            frame=self.dict[f]
            image=frame.rightimg
            if image!=None:
                sp=image.shape
                if sp[0]>sp[1]:
                    img2=cv2.copyMakeBorder(image, 0,0, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2), cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
                else:
                    img2=cv2.copyMakeBorder(image, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2),0,0, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
                img3=cv2.resize(img2,(128,128))
                image=img3
                image = color.rgb2gray(image)
                fd, hog_image = hog(image, orientations=9, pixels_per_cell=(32,32),cells_per_block=(2, 2), visualise=True)

                hogSet.append(fd)'''




        '''pca = RandomizedPCA(n_components=51)

        pca.fit(hogSet)
        hogSet = pca.transform(hogSet).tolist()'''
        hogSet=hogSet.tolist()
        avehog=self.pooling(hogSet,1)
        cnt=0
        max=float('inf')
        key=0
        for f in hogSet:
            dis=0
            #self.dict[f].hog=hogSet[cnt]

            for i in range(51):
                dis+=(avehog[i]-hogSet[cnt][i])**2
            if dis<max:
                max=dis
                key=cnt
            cnt+=1
        self.hogFeature.append(hogSet[key])
    def consTrajectoryYin(self,frames,mode):
        if mode=='microsoft':
            pointNo=60
        else:
            pointNo=10
        data=[]

        frameno=0
        for f in frames:
            if self.dict[f].position==[0,0,0,0]:
                continue
            frameno+=1
            if self.dict[f].data!=[]:
                data.append(self.dict[f].data)

        self.xtrajectory=[]
        self.ytrajectory=[]
        self.ztrajectory=[]
        self.extrajectory=[]
        self.eytrajectory=[]
        self.eztrajectory=[]
        self.xlefttrajectory=[]
        self.ylefttrajectory=[]
        self.zlefttrajectory=[]
        self.exlefttrajectory=[]
        self.eylefttrajectory=[]
        self.ezlefttrajectory=[]




        if data==[]:
            self.xtrajectory=[0]*10
            self.ytrajectory=[0]*10
            self.ztrajectory=[0]*10
            self.extrajectory=[0]*10
            self.eytrajectory=[0]*10
            self.eztrajectory=[0]*10
            self.xlefttrajectory=[0]*10
            self.ylefttrajectory=[0]*10
            self.zlefttrajectory=[0]*10
            self.exlefttrajectory=[0]*10
            self.eylefttrajectory=[0]*10
            self.ezlefttrajectory=[0]*10
            return
        if len(data)==1 or len(data)==2:
            self.xtrajectory=[data[0][-3]]*10
            self.ytrajectory=[data[0][-2]]*10
            self.ztrajectory=[data[0][-1]]*10
            self.extrajectory=[data[0][9]]*10
            self.eytrajectory=[data[0][10]]*10
            self.eztrajectory=[data[0][11]]*10
            self.xlefttrajectory=[data[0][-4]]*10
            self.ylefttrajectory=[data[0][-5]]*10
            self.zlefttrajectory=[data[0][-6]]*10
            self.exlefttrajectory=[data[0][6]]*10
            self.eylefttrajectory=[data[0][7]]*10
            self.ezlefttrajectory=[data[0][8]]*10
            return
        totalDis=0
        #etotalDis=0
        for i in range(len(data)-1):
            if data[i]==[]:
                continue
            totalDis+=math.sqrt((data[i+1][-1]-data[i][-1])**2+(data[i+1][-2]-data[i][-2])**2+(data[i+1][-3]-data[i][-3])**2)
            #etotalDis+=math.sqrt((data[i+1][9]-data[i][9])**2+(data[i+1][10]-data[i][10])**2+(data[i+1][11]-data[i][11])**2)


        step0=totalDis/pointNo


        cuframe=0
        framed=0
        for i in range(pointNo):
            if cuframe==len(data)-1 or cuframe>len(data)-1:
                while len(self.xtrajectory)<pointNo:
                    self.xtrajectory.append(data[len(data)-1][-3])
                    self.ytrajectory.append(data[len(data)-1][-2])
                    self.ztrajectory.append(data[len(data)-1][-1])
                break

            framedis=math.sqrt((data[cuframe+1][-3]-data[cuframe][-3])**2+(data[cuframe+1][-2]-data[cuframe][-2])**2+(data[cuframe+1][-1]-data[cuframe][-1])**2)
            while framedis==0:
                cuframe+=1
                if cuframe<len(data)-1:
                    framedis=math.sqrt((data[cuframe+1][-3]-data[cuframe][-3])**2+(data[cuframe+1][-2]-data[cuframe][-2])**2+(data[cuframe+1][-1]-data[cuframe][-1])**2)
                else:
                    return 0
            framedis-=framed
            step=step0
            while(step>framedis):
                step-=framedis
                cuframe+=1
                framed=0
                if cuframe<len(data)-1:
                    framedis=math.sqrt((data[cuframe+1][-3]-data[cuframe][-3])**2+(data[cuframe+1][-2]-data[cuframe][-2])**2+(data[cuframe+1][-1]-data[cuframe][-1])**2)
                else:
                    self.xtrajectory.append(data[len(data)-1][-3])
                    self.ytrajectory.append(data[len(data)-1][-2])
                    self.ztrajectory.append(data[len(data)-1][-1])

                    break
            if step<=framedis and cuframe<len(data)-1:
                #print data[cuframe+1]
                #print data[cuframe]


                fraction=(step+framed)/math.sqrt((data[cuframe+1][-1]-data[cuframe][-1])**2+(data[cuframe+1][-2]-data[cuframe][-2])**2+(data[cuframe+1][-3]-data[cuframe][-3])**2)
                x=fraction*data[cuframe+1][-3]+(1-fraction)*data[cuframe][-3]
                y=fraction*data[cuframe+1][-2]+(1-fraction)*data[cuframe][-2]
                z=fraction*data[cuframe+1][-1]+(1-fraction)*data[cuframe][-1]
                #xi=(float(x)-headx)/shoulder
                #yi=(float(y)-heady)/tall
                self.xtrajectory.append(x)
                self.ytrajectory.append(y)
                self.ztrajectory.append(z)

                framed+=step

        while len(self.xtrajectory)<pointNo:
            self.xtrajectory.append(data[len(data)-1][-3])
            self.ytrajectory.append(data[len(data)-1][-2])
            self.ztrajectory.append(data[len(data)-1][-1])


        #right elbow
        assert self.xtrajectory!=[]
        totalDis=0
        for i in range(len(data)-1):
            if data[i]==[]:
                continue
            totalDis+=math.sqrt((data[i+1][9]-data[i][9])**2+(data[i+1][10]-data[i][10])**2+(data[i+1][11]-data[i][11])**2)
        step0=totalDis/pointNo
        cuframe=0
        framed=0
        for i in range(pointNo):
            if cuframe==len(data)-1 or cuframe>len(data)-1:
                while len(self.extrajectory)<pointNo:
                    self.extrajectory.append(data[-1][9])
                    self.eytrajectory.append(data[-1][10])
                    self.eztrajectory.append(data[-1][11])
                break
            framedis=math.sqrt((data[cuframe+1][9]-data[cuframe][9])**2+(data[cuframe+1][10]-data[cuframe][10])**2+(data[cuframe+1][11]-data[cuframe][11])**2)
            while framedis==0:
                cuframe+=1
                if cuframe<len(data)-1:
                    framedis=math.sqrt((data[cuframe+1][9]-data[cuframe][9])**2+(data[cuframe+1][10]-data[cuframe][10])**2+(data[cuframe+1][11]-data[cuframe][11])**2)
                else:
                    return 0
            framedis-=framed
            step=step0
            while(step>framedis):
                step-=framedis
                cuframe+=1
                framed=0
                if cuframe<len(data)-1:
                    framedis=math.sqrt((data[cuframe+1][9]-data[cuframe][9])**2+(data[cuframe+1][10]-data[cuframe][10])**2+(data[cuframe+1][11]-data[cuframe][11])**2)
                else:
                    self.extrajectory.append(data[len(data)-1][9])
                    self.eytrajectory.append(data[len(data)-1][10])
                    self.eztrajectory.append(data[len(data)-1][11])
                    break
            if step<=framedis and cuframe<len(data)-1:
                #print data[cuframe+1]
                #print data[cuframe]


                fraction=(step+framed)/math.sqrt((data[cuframe+1][9]-data[cuframe][9])**2+(data[cuframe+1][10]-data[cuframe][10])**2+(data[cuframe+1][11]-data[cuframe][11])**2)
                x=fraction*data[cuframe+1][9]+(1-fraction)*data[cuframe][9]
                y=fraction*data[cuframe+1][10]+(1-fraction)*data[cuframe][10]
                z=fraction*data[cuframe+1][11]+(1-fraction)*data[cuframe][11]
                #xi=(float(x)-headx)/shoulder
                #yi=(float(y)-heady)/tall
                self.extrajectory.append(x)
                self.eytrajectory.append(y)
                self.eztrajectory.append(z)

                framed+=step

        while len(self.extrajectory)<pointNo:
            self.extrajectory.append(data[len(data)-1][9])
            self.eytrajectory.append(data[len(data)-1][10])
            self.eztrajectory.append(data[len(data)-1][11])





        #left hand












        if data==[] or len(data)==1:
            return 0
        totalDis=0
        for i in range(len(data)-1):
            if data[i]==[]:
                continue
            totalDis+=math.sqrt((data[i+1][-4]-data[i][-4])**2+(data[i+1][-5]-data[i][-5])**2+(data[i+1][-6]-data[i][-6])**2)

        step0=totalDis/pointNo

        step=step0
        cuframe=0
        #steped=0
        framed=0
        newframe=1
        for i in range(pointNo):
            if cuframe==len(data)-1 or cuframe>len(data)-1:
                while len(self.xlefttrajectory)<pointNo:
                    self.xlefttrajectory.append(data[len(data)-1][-4])
                    self.ylefttrajectory.append(data[len(data)-1][-5])
                    self.zlefttrajectory.append(data[len(data)-1][-6])


                break
            framedis=math.sqrt((data[cuframe+1][-4]-data[cuframe][-4])**2+(data[cuframe+1][-5]-data[cuframe][-5])**2+(data[cuframe+1][-6]-data[cuframe][-6])**2)
            while framedis==0:
                cuframe+=1
                if cuframe<len(data)-1:
                    framedis=math.sqrt((data[cuframe+1][-4]-data[cuframe][-4])**2+(data[cuframe+1][-5]-data[cuframe][-5])**2+(data[cuframe+1][-6]-data[cuframe][-6])**2)
                else:
                    while self.xlefttrajectory<pointNo:
                        self.xlefttrajectory.append(data[len(data)-1][-4])
                        self.ylefttrajectory.append(data[len(data)-1][-5])
                        self.zlefttrajectory.append(data[len(data)-1][-6])


            framedis-=framed
            step=step0
            while(step>framedis):
                step-=framedis
                cuframe+=1
                framed=0
                if cuframe<len(data)-1:
                        framedis=math.sqrt((data[cuframe+1][-4]-data[cuframe][-4])**2+(data[cuframe+1][-5]-data[cuframe][-5])**2+(data[cuframe+1][-6]-data[cuframe][-6])**2)
                else:
                        break
            if step<=framedis and cuframe<len(data)-1:
                fraction=(step+framed)/math.sqrt((data[cuframe+1][-4]-data[cuframe][-4])**2+(data[cuframe+1][-5]-data[cuframe][-5])**2+(data[cuframe+1][-6]-data[cuframe][-6])**2)
                x=fraction*data[cuframe+1][-4]+(1-fraction)*data[cuframe][-4]
                y=fraction*data[cuframe+1][-5]+(1-fraction)*data[cuframe][-5]
                z=fraction*data[cuframe+1][-6]+(1-fraction)*data[cuframe][-6]

                self.xlefttrajectory.append(x)
                self.ylefttrajectory.append(y)
                self.zlefttrajectory.append(z)


                framed+=step
        while len(self.xlefttrajectory)<pointNo:
            self.xlefttrajectory.append(data[len(data)-1][-4])
            self.ylefttrajectory.append(data[len(data)-1][-5])
            self.zlefttrajectory.append(data[len(data)-1][-6])








        #left elbow




        if data==[] or len(data)==1:
            return 0
        totalDis=0
        for i in range(len(data)-1):
            if data[i]==[]:
                continue
            totalDis+=math.sqrt((data[i+1][6]-data[i][6])**2+(data[i+1][7]-data[i][7])**2+(data[i+1][8]-data[i][8])**2)

        step0=totalDis/pointNo

        step=step0
        cuframe=0
        #steped=0
        framed=0
        newframe=1
        for i in range(pointNo):
            if cuframe==len(data)-1 or cuframe>len(data)-1:
                while len(self.exlefttrajectory)<pointNo:
                    self.exlefttrajectory.append(data[len(data)-1][6])
                    self.eylefttrajectory.append(data[len(data)-1][7])
                    self.ezlefttrajectory.append(data[len(data)-1][8])


                break
            framedis=math.sqrt((data[cuframe+1][6]-data[cuframe][6])**2+(data[cuframe+1][7]-data[cuframe][7])**2+(data[cuframe+1][8]-data[cuframe][8])**2)
            while framedis==0:
                cuframe+=1
                if cuframe<len(data)-1:
                    framedis=math.sqrt((data[cuframe+1][6]-data[cuframe][6])**2+(data[cuframe+1][7]-data[cuframe][7])**2+(data[cuframe+1][8]-data[cuframe][8])**2)
                else:
                    while self.exlefttrajectory<pointNo:
                        self.exlefttrajectory.append(data[len(data)-1][6])
                        self.eylefttrajectory.append(data[len(data)-1][7])
                        self.ezlefttrajectory.append(data[len(data)-1][8])


            framedis-=framed
            step=step0
            while(step>framedis):
                step-=framedis
                cuframe+=1
                framed=0
                if cuframe<len(data)-1:
                        framedis=math.sqrt((data[cuframe+1][6]-data[cuframe][6])**2+(data[cuframe+1][7]-data[cuframe][7])**2+(data[cuframe+1][8]-data[cuframe][8])**2)
                else:
                        break
            if step<=framedis and cuframe<len(data)-1:
                fraction=(step+framed)/math.sqrt((data[cuframe+1][6]-data[cuframe][6])**2+(data[cuframe+1][7]-data[cuframe][7])**2+(data[cuframe+1][8]-data[cuframe][8])**2)
                x=fraction*data[cuframe+1][6]+(1-fraction)*data[cuframe][6]
                y=fraction*data[cuframe+1][7]+(1-fraction)*data[cuframe][7]
                z=fraction*data[cuframe+1][8]+(1-fraction)*data[cuframe][8]

                self.exlefttrajectory.append(x)
                self.eylefttrajectory.append(y)
                self.ezlefttrajectory.append(z)


                framed+=step
        while len(self.exlefttrajectory)<pointNo:
            self.exlefttrajectory.append(data[len(data)-1][6])
            self.eylefttrajectory.append(data[len(data)-1][7])
            self.ezlefttrajectory.append(data[len(data)-1][8])


    def buildFeatureYin(self):
        l=math.ceil(len(self.framelistYin)/5)

        self.combinedFeature=[]
        print self.path
        for i in range(5):
            frames=[]
            for f in range(self.framelistYin[int(l*i)],self.framelistYin[min(len(self.framelistYin)-1,int(l*(i+1)))]):
                if f in self.framelistYin:
                    frames.append(f)

            self.consTrajectoryYin(frames,'')
            #self.getHogYin(frames)
            trajectory=self.xtrajectory+self.ytrajectory+self.ztrajectory+self.extrajectory+self.eytrajectory+self.eztrajectory+self.xlefttrajectory+self.ylefttrajectory+self.zlefttrajectory+self.exlefttrajectory+self.eylefttrajectory+self.ezlefttrajectory

            while len(self.hogFeature)<5:
                self.hogFeature.append(self.hogFeature[-1])

            self.combinedFeature.append(trajectory+self.hogFeature[i])

            #assert trajectory+self.hogFeature[i]==171
class YinClassifier(Classifier):
    def pcafeature(self):
        feature1=[]
        feature2=[]
        feature3=[]
        for path in self.filelist:

            if self.dic[path].bothseparate==0 and self.dic[path].intersect==0:
                feature1.append(normalize_histogram(self.dic[path].handshape)+normalize_histogram(self.dic[path].hogFeature)+normalize_histogram_abs(self.dic[path].xtrajectory+self.dic[path].ytrajectory))


            elif self.dic[path].bothseparate==1:
                feature2.append(normalize_histogram(self.dic[path].handshape)+normalize_histogram(self.dic[path].hogFeature)+normalize_histogram_abs(self.dic[path].xtrajectory+self.dic[path].ytrajectory))

            elif self.dic[path].intersect==1:
                feature3.append(normalize_histogram(self.dic[path].handshape)+normalize_histogram(self.dic[path].hogFeature)+normalize_histogram_abs(self.dic[path].xtrajectory+self.dic[path].ytrajectory))


        pca = PCA(n_components=200)
        pca.fit(feature1)
        feature1 = pca.transform(feature1).tolist()
        pca = PCA(n_components=200)
        pca.fit(feature2)
        feature2 = pca.transform(feature2).tolist()
        pca = PCA(n_components=200)
        pca.fit(feature3)
        feature3 = pca.transform(feature3).tolist()

        for path in self.filelist:
            if self.dic[path].bothseparate==0 and self.dic[path].intersect==0:
                self.dic[path].combinedFeature=feature1.pop()

        for path in self.filelist:
            if self.dic[path].bothseparate==1:
                self.dic[path].combinedFeature=feature2.pop()
        for path in self.filelist:
            if self.dic[path].intersect==1:
                self.dic[path].combinedFeature=feature3.pop()
        assert feature1==[] and feature2==[] and feature3==[]


    def buildFeatureYin(self):
        name=[]
        part=[]
        for path in self.filelist:
            self.dic[path].buildFeatureYin()
            name.append(self.dic[path].wordName)
            name.append(self.dic[path].wordName)
            name.append(self.dic[path].wordName)
            name.append(self.dic[path].wordName)
            name.append(self.dic[path].wordName)
            part.append(1)
            part.append(2)
            part.append(3)
            part.append(4)
            part.append(5)
        dict={}
        features=[]
        dict['w']=np.zeros((len(self.filelist)*5,len(self.filelist)*5))


        for path in self.filelist:
            print path
            for i in range(5):
                features.append(self.dic[path].combinedFeature[i])
                assert len(self.dic[path].combinedFeature[i])==171
        x=0
        for path in self.filelist:
            for i in range(5):
                y=0
                for path2 in self.filelist:
                    for k in range(5):
                        if self.dic[path].wordName==self.dic[path2].wordName and i%5==k%5:
                            dict['w'][x][y]=1
                        y+=1
                x+=1


        dict['x']=np.asarray(features)
        #print dict['x'].shape
        #for i in range(len(dict['x'])):
        #    assert len(features[i])==171
        #print dict['w']
        assert len(features)%5==0
        scipy.io.savemat('/home/lzz/Aaron.mat', dict)


    def buildpca(self):
        totalset=[]
        for path in self.filelist:
            l=math.ceil(len(self.dic[path].framelistYin)/5)
            for i in range(5):
                frames=[]
                for f in range(self.dic[path].framelistYin[int(l*i)],self.dic[path].framelistYin[min(len(self.dic[path].framelistYin)-1,int(l*(i+1)))]):
                    if f in self.dic[path].framelistYin:
                        frames.append(f)
                if self.dic[path].traintest=='train':
                    hogset=self.dic[path].buildpca(frames)
                    for hog in hogset:
                        totalset.append(hog)
                if self.dic[path].traintest=='test':
                    self.dic[path].buildpca(frames)
        pca = PCA(n_components=51)


        pca.fit(totalset)
        #hogSet = pca.transform(totalset).tolist()
        for path in self.filelist:
            print path
            for i in range(5):
                print pca.transform(self.dic[path].hogset[i]).shape
                self.dic[path].getHogYin(pca.transform(self.dic[path].hogset[i]))
if __name__ == '__main__':
    classifier = Classifier()



    dataset='our'

    trainname={}
    testname={}

    if dataset=='devisign':
        pathTotal='/media/lzz/Data1/devisign/'
        trainname['P08']=0
        trainname['P02']=0
        trainname['P01']=0
        trainname['P07']=0
        testname['P03']=0
        classifier.listdevisign(pathTotal)
        classifier.splitdevisign(trainname,testname,'',0)

    elif dataset=='our':

        #pathTotal='/media/lzz/Data1/kinect/'
        #pathTotal='/home/lzz/sign/data1/'
        pathTotal='/media/lzz/HD1/1Michael/split/new/'
        trainname['lzz']=0
        trainname['hfy']=0
        trainname['fuyang']=0
        testname['Aaron']=0
        testname['Michael']=0
        testname['Micheal']=0
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



    mode=2


    wholeMode=1
    classifier.buildpca()

    classifier.buildFeatureYin()
