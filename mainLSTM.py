import lstm.RNN_with_gating

#import whole_network,whole_level_network
import caffeDL
from Classifier import *
class lstmClassifier(Classifier):
    def preLSTM(self):
        train1=[]
        target1=[]
        train2=[]
        target2=[]
        train3=[]
        target3=[]

        test1=[]
        testtarget1=[]
        test2=[]
        testtarget2=[]
        test3=[]
        testtarget3=[]

        t1=0
        t2=0
        t3=0
        set1={}
        set2={}
        set3={}


        for path in self.filelist:
            print path
            if(self.dic[path].bothseparate==0 and self.dic[path].intersect==0):
                if (set1.has_key(self.dic[path].wordName))==0:
                    set1[self.dic[path].wordName]=t1
                    t1+=1
                data=[]
                #label=[]
                for i in range(len(self.dic[path].xtrajectory)):
                    data.append([])
                    data[i].append(self.dic[path].xtrajectory[i])
                    data[i].append(self.dic[path].ytrajectory[i])
                if(self.dic[path].traintest=='train'):
                    train1.append(data)
                    target1.append(set1[self.dic[path].wordName])
                elif(self.dic[path].traintest=='test'):
                    test1.append(data)
                    testtarget1.append(set1[self.dic[path].wordName])


            if(self.dic[path].bothseparate==1):
                if (set2.has_key(self.dic[path].wordName))==0:
                    set2[self.dic[path].wordName]=t2
                    t2+=1
                data=[]
                #label=[]
                for i in range(len(self.dic[path].xtrajectory)):
                    data.append([])
                    data[i].append(self.dic[path].xtrajectory[i])
                    data[i].append(self.dic[path].ytrajectory[i])
                    data[i].append(self.dic[path].xlefttrajectory[i])
                    data[i].append(self.dic[path].ylefttrajectory[i])
                if(self.dic[path].traintest=='train'):
                    train2.append(data)
                    target2.append(set2[self.dic[path].wordName])
                elif(self.dic[path].traintest=='test'):
                    test2.append(data)
                    testtarget2.append(set2[self.dic[path].wordName])

            if(self.dic[path].intersect==1):
                if (set3.has_key(self.dic[path].wordName))==0:
                    set3[self.dic[path].wordName]=t3
                    t3+=1
                data=[]

                for i in range(len(self.dic[path].xtrajectory)):
                    data.append([])
                    data[i].append(self.dic[path].xtrajectory[i])
                    data[i].append(self.dic[path].ytrajectory[i])
                if(self.dic[path].traintest=='train'):

                    train3.append(data)
                    target3.append(set3[self.dic[path].wordName])
                elif(self.dic[path].traintest=='test'):

                    test3.append(data)
                    testtarget3.append(set3[self.dic[path].wordName])
        self.name2labellstm1=set1
        self.name2labellstm2=set2
        self.name2labellstm3=set3
        return [[train1,train2,train3],[target1,target2,target3],[t1,t2,t3]],[[test1,test2,test3],[testtarget1,testtarget2,testtarget3],[t1,t2,t3]]

    def prewhole(self):
        set1={}
        set2={}
        set3={}
        t1=0
        t2=0
        t3=0

        for path in self.filelist:
            print path
            if(self.dic[path].bothseparate==0 and self.dic[path].intersect==0):
                if (set1.has_key(self.dic[path].wordName))==0:
                    set1[self.dic[path].wordName]=t1
                    t1+=1


            if(self.dic[path].bothseparate==1):
                if (set2.has_key(self.dic[path].wordName))==0:
                    set2[self.dic[path].wordName]=t2
                    t2+=1

            if(self.dic[path].intersect==1):
                if (set3.has_key(self.dic[path].wordName))==0:
                    set3[self.dic[path].wordName]=t3
                    t3+=1

        self.name2labellstm1=set1
        self.name2labellstm2=set2
        self.name2labellstm3=set3
         #train, valid, test = imdb.load_data(train1,test1,n_words=n_words, valid_portion=0.05,
        #                       maxlen=maxlen)

        #lstm.train_lstm(train1,test1,max_epochs=100,saveto='lstm_model_single.npz',test_size=500)
        #lstm.train_lstm(train2,test2,max_epochs=100,saveto='lstm_model_both.npz',test_size=500)
        #lstm.train_lstm(train3,test3,max_epochs=100,saveto='lstm_model_inter.npz',test_size=500)

    def trainLSTM(self):
        train1=[]
        target1=[]
        train2=[]
        target2=[]
        train3=[]
        target3=[]

        test1=[]
        testtarget1=[]
        test2=[]
        testtarget2=[]
        test3=[]
        testtarget3=[]

        t1=0
        t2=0
        t3=0
        set1={}
        set2={}
        set3={}
        for path in self.filelist:
            print path
            if(self.dic[path].bothseparate==0 and self.dic[path].intersect==0):
                if (set1.has_key(self.dic[path].wordName))==0:
                    set1[self.dic[path].wordName]=t1
                    t1+=1
                data=[]
                #label=[]
                for i in range(len(self.dic[path].xtrajectory)):
                    data.append([])
                    data[i].append(self.dic[path].xtrajectory[i])
                    data[i].append(self.dic[path].ytrajectory[i])
                if(self.dic[path].traintest=='train'):
                    train1.append(data)
                    target1.append(set1[self.dic[path].wordName])
                elif(self.dic[path].traintest=='test'):
                    test1.append(data)
                    testtarget1.append(set1[self.dic[path].wordName])


            if(self.dic[path].bothseparate==1):
                if (set2.has_key(self.dic[path].wordName))==0:
                    set2[self.dic[path].wordName]=t2
                    t2+=1
                data=[]
                #label=[]
                for i in range(len(self.dic[path].xtrajectory)):
                    data.append([])
                    data[i].append(self.dic[path].xtrajectory[i])
                    data[i].append(self.dic[path].ytrajectory[i])
                    data[i].append(self.dic[path].xlefttrajectory[i])
                    data[i].append(self.dic[path].ylefttrajectory[i])
                if(self.dic[path].traintest=='train'):
                    train2.append(data)
                    target2.append(set2[self.dic[path].wordName])
                elif(self.dic[path].traintest=='test'):
                    test2.append(data)
                    testtarget2.append(set2[self.dic[path].wordName])

            if(self.dic[path].intersect==1):
                if (set3.has_key(self.dic[path].wordName))==0:
                    set3[self.dic[path].wordName]=t3
                    t3+=1
                data=[]

                for i in range(len(self.dic[path].xtrajectory)):
                    data.append([])
                    data[i].append(self.dic[path].xtrajectory[i])
                    data[i].append(self.dic[path].ytrajectory[i])
                if(self.dic[path].traintest=='train'):

                    train3.append(data)
                    target3.append(set3[self.dic[path].wordName])
                elif(self.dic[path].traintest=='test'):

                    test3.append(data)
                    testtarget3.append(set3[self.dic[path].wordName])

        lstm.RNN_with_gating.train_softmax(np.asarray(train1,dtype='float32'),np.asarray(target1),np.asarray(test1),np.asarray(testtarget1),n_y=t1,type="single")
        lstm.RNN_with_gating.train_softmax(np.asarray(train2,dtype='float32'),np.asarray(target2),np.asarray(test2),np.asarray(testtarget2),n_y=t2,type="both")
        lstm.RNN_with_gating.train_softmax(np.asarray(train3,dtype='float32'),np.asarray(target3),np.asarray(test3),np.asarray(testtarget3),n_y=t3,type="intersect")


    def testLSTM(self,test1,test2,test3,testtarget1,testtarget2,testtarget3,t1,t2,t3):

        model1="/home/lzz/project/project/lstm/model/picklemodel"
        #model2=np.load('/home/lzz/project/project/lstm/model/both3000.npz')['param']
        #model3=np.load('/home/lzz/project/project/lstm/model/intersect3000.npz')['param']
        lstm.RNN_with_gating.test_softmax(model1,np.asarray(test1),np.asarray(testtarget1),t1)
        #lstm.RNN_with_gating.test_softmax(model2,np.asarray(test2),np.asarray(testtarget2))
        #lstm.RNN_with_gating.test_softmax(model3,np.asarray(test3),np.asarray(testtarget3))

    def getEle(self,path):
        batch=[]
        if self.dic[path].intersect==0:

            for i in range(self.dic[path].keyNo):

                img=self.dic[path].dict[self.dic[path].topIndex[i]].rightimg
                sp=img.shape
                if sp[0]>sp[1]:
                    img2=cv2.copyMakeBorder(img, 0,0, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2), cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
                else:
                    img2=cv2.copyMakeBorder(img, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2),0,0, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
                img3=cv2.resize(img2,(128,128))
                batch.append(img3[:,:,0].reshape(128*128))

        data=[]
        if(self.dic[path].bothseparate==1):


            #label=[]
            for i in range(len(self.dic[path].xtrajectory)):
                data.append([])
                data[i].append(self.dic[path].xtrajectory[i])
                data[i].append(self.dic[path].ytrajectory[i])

        target=[self.name2labellstm1[self.dic[path].wordName]]

        return batch,data,target,len(self.dic)

    def getEleLevel(self,path):
        batch=[]
        #if self.dic[path].intersect==0:

        for i in range(self.dic[path].keyNo):

            img=self.dic[path].dict[self.dic[path].topIndex[i]].rightimg
            sp=img.shape
            if sp[0]>sp[1]:
                img2=cv2.copyMakeBorder(img, 0,0, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2), cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
            else:
                img2=cv2.copyMakeBorder(img, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2),0,0, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
            img3=cv2.resize(img2,(128,128))
            batch.append(img3[:,:,0].reshape(128*128))

        data=self.dic[path].topPosition

        while len(batch)<10:
            batch.append(batch[-1])
        while len(data)<10:
            data.append(data[-1])


        if self.dic[path].bothseparate==0 and self.dic[path].intersect==0:
            target=[self.name2labellstm1[self.dic[path].wordName]]
        elif self.dic[path].bothseparate==1:
            target=[self.name2labellstm2[self.dic[path].wordName]]
        elif self.dic[path].intersect==1:
            target=[self.name2labellstm3[self.dic[path].wordName]]

        return batch,data,target,len(self.dic)

    '''def trainWholeCombineFeature(self,n_epochs=100):
        deepl=whole_level_network.level_network()
        deepl.build_whole_network()

        deepl.cn.load_params('/home/lzz/project/project/cnn/model/picklemodel15')
        deepl.rn.load_params('/home/lzz/project/project/lstm/model/picklemodel')
        patience = 1000  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is
                               # found
        improvement_threshold = 0.995  # a relative improvement of this much is
                                       # considered significant
        best_validation_loss = numpy.inf
        best_iter = 0
        test_score = 0.
        start_time = time.clock()

        epoch = 0
        test_set=[]
        while (epoch < n_epochs):
            epoch = epoch + 1
            for path in self.filelist:
                print path
                batch,data,target,pathNo=self.getEle(path)
                if self.dic[path].traintest=='train':
                    deepl.train(batch,data,target)
                else:
                    test_set.append([batch,data,target])
            if epoch%10==0:
                test_losses=[]
                for test_sample in test_set:
                    test_losses.append(deepl.test(test_sample[0],test_sample[1],test_sample[2]))
                test_score = numpy.mean(test_losses)
                print 'test_score'
                print test_score'''


    def trainWholeLevelFeature(self,n_epochs=100):
        deepl=whole_level_network.level_network()

        deepl.build_whole_network(50)

        deepl.rn.cn.load_params('/home/lzz/sign/model/picklemodel13')
        epoch = 0
        test_set=[]
        while (epoch < n_epochs):
            epoch = epoch + 1
            cost=0
            num=0
            for path in self.filelist:
                print path
                batch,data,target,pathNo=self.getEleLevel(path)
                #print '####'
                #print len(batch),batch[0].shape
                #print len(data),len(data[0])
                #print data
                #print target
                #print self.dic[path].traintest
                #print 'data size'
                #print numpy.asarray(data).shape
                #print np.asarray(batch).shape
                #print np.asarray([data]).shape
                #print path,self.dic[path].intersect,self.dic[path].shouldgenerate,self.dic[path].bothseparate,self.dic[path].traintest
                if self.dic[path].traintest=='train':
                    costi=deepl.train(batch,data,target)
                    cost+=costi
                    num+=1
                else:
                    test_set.append([batch,data,target])
            print 'cost:',cost/num
            if epoch%10==0:
                guesses=[]
                test_losses=[]
                for test_sample in test_set:
                    #print test_sample[0][1].shape
                    guess,loss=deepl.test(test_sample[0],test_sample[1],test_sample[2])
                    #print guess
                    guesses.append(guess)
                    test_losses.append(loss)
                    #print str(test_sample[2])+' '+str(guess)
                test_score = numpy.mean(test_losses)
                print 'test_score'
                print test_score
                r=0
                w=0
                print numpy.asarray(guesses).shape
                print numpy.asarray(test_set).shape
                for i in range(len(guesses)):
                    if guesses[i]==test_set[i][2]:
                        r+=1
                    else:
                        w+=1
                print 'accuracy='+str(r/(r+w))

                deepl.rn.save_params(deepl.rn.get_params(), "/home/lzz/sign/model/level/levelmodel"+str(epoch))

if __name__ == '__main__':
    caffedl=caffeDL('/home/lzz/sign/proto/lenet_test.prototxt','/home/lzz/sign/model/lenet_iter_5000.caffemodel')
    #caffedl=caffeDL('/home/lzz/caffe/caffe-master/proto/lenet_test.prototxt','/home/lzz/sign/200d/lenet_iter_1800.caffemodel')
    caffedlInter=caffeDL('/home/lzz/sign/proto_inter/lenet_test.prototxt','/home/lzz/sign/model/lenet__iter_400.caffemodel')
    #caffedlInter=caffeDL('/home/lzz/sign/project/proto_inter/lenet_test.prototxt','/home/lzz/sign/project/model/inter_2/lenet__iter_300.caffemodel')
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
        pathTotal='/media/lzz/Data1/kinect/similardata/'
        #pathTotal='/media/lzz/HD1/newkinect/'
        trainname['hfy']=0
        trainname['fuyang']=0

        trainname['lzz']=0
        testname['Aaron']=0
        #classifier.listFile(pathTotal)
        #classifier.split(trainname,testname,'xn',3)
        #classifier.split(trainname,testname,'lzz',1)
        dic={}
        dic['Aaron']=0
        dic['Michael']=0
        dic['Micheal']=0
        classifier.listFile(pathTotal)
        classifier.split(trainname,testname,dic,0)

    classifier.constructLabelData()
    classifier.label2Name={}
    for path in classifier.filelist:
        classifier.label2Name[classifier.dic[path].label]=classifier.dic[path].wordName




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

    #classifier.getHogFeature()


    [[train1,train2,train3],[target1,target2,target3],[t1,t2,t3]],[[test1,test2,test3],[testtarget1,testtarget2,testtarget3],[t1,t2,t3]]=classifier.preLSTM()
    classifier.trainWholeCombineFeature()

    classifier.trainWhole()
    classifier.prewhole()
    classifier.trainWholeLevelFeature()

    classifier.trainLSTM()
    classifier.testLSTM(test1,test2,test3,testtarget1,testtarget2,testtarget3,t1,t2,t3)
