from Classifier import *
class MicrosoftSignWord(SignWord):
    def combineFeatureMicrosoft(self):
        print self.path
        '''self.xtrajectory=normalize_histogram(self.xtrajectory)
        self.ytrajectory=normalize_histogram(self.ytrajectory)
        self.ztrajectory=normalize_histogram(self.ztrajectory)
        self.xlefttrajectory=normalize_histogram(self.xlefttrajectory)
        self.ylefttrajectory=normalize_histogram(self.ylefttrajectory)
        self.zlefttrajectory=normalize_histogram(self.zlefttrajectory)'''
        self.combinedFeature=[[],[],[],[],[],[]]
        for i in range(len(self.xtrajectory)):

            self.combinedFeature[0].append(self.xtrajectory[i])
            self.combinedFeature[1].append(self.ytrajectory[i])
            self.combinedFeature[2].append(self.ztrajectory[i])
            self.combinedFeature[3].append(self.xlefttrajectory[i])
            self.combinedFeature[4].append(self.ylefttrajectory[i])
            self.combinedFeature[5].append(self.zlefttrajectory[i])
        del self.dict
        self.alsoIntersectSet=[]
        self.handshape=[]
        self.displacement=[]
        self.lefthandshape=[]
        self.trajectory=[]
class MicrosoftClassifier(Classifier):

    def recogMicrosoft(self):
        features={}

        featuresboth={}

        for path in self.trainSet:

            features[self.dic[path].path]=self.dic[path].combinedFeature




        correct=0
        wrong=0

        for path in self.testSet:
            prediction=self.dic[path].wordName


            dist=float("inf")
            for p in features:
                tmpdist=dtw(np.asarray(self.dic[path].combinedFeature),np.asarray(self.dic[p].combinedFeature))[0]
                if tmpdist<dist:
                    dist=tmpdist
                    prediction=self.dic[p].wordName
            if self.dic[path].wordName==prediction:
                correct+=1
            else:
                wrong+=1
        print correct,wrong
        accuracy=float(correct)/float(correct+wrong)
        print accuracy

    def constructTrajectoryMicrosoft(self):
        for path in self.filelist:
            frames=self.dic[path].framelist
            self.dic[path].consTrajectoryYin(frames,'microsoft')
    def combineFeatureMicrosoft(self):
        for path in self.filelist:
            self.dic[path].combineFeatureMicrosoft()
#Microsoft
if __name__ == '__main__':
    classifier = Classifier()
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
        #pathTotal='/home/lzz/sign/data/'
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
        #print len(classifier.filelist)
        classifier.split(trainname,testname,dic,2)

    classifier.constructLabelData()
    classifier.label2Name={}
    for path in classifier.filelist:
        classifier.label2Name[classifier.dic[path].label]=classifier.dic[path].wordName




    mode=2


    wholeMode=1



    classifier.constructTrajectoryMicrosoft()




    classifier.combineFeatureMicrosoft()
    classifier.recogMicrosoft()




