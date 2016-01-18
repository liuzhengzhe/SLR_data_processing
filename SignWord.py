import glob
import csv
import cv2
import numpy as np
import math
from frame import frame
from normalize import *
import hogmodule
from skimage.feature import hog
from skimage import data, color, exposure
import os
import numpy
import shutil
from scipy import spatial
import h5py
class SignWord():
    def __init__(self,path,datamode):
        loc=path.rfind("/")
        name=path[loc+1:]
        self.sampleName=name
        if datamode==0:
            self.wordName=self.sampleName[0:self.sampleName.find(" ")]
            self.signer=self.sampleName[self.sampleName.find(" ")+1:self.sampleName.find(" ",self.sampleName.find(" ")+1)]
        elif datamode==1:
            self.wordName=self.sampleName.split('_')[1]
            self.signer=self.sampleName.split('_')[0]
        self.displacement=[]
        self.handshape=[]
        self.hogFeature=[]
        self.single2=0
        self.intersect=0
        self.bothseparate=0
        self.keyNo=10
        self.leftkeyNo=0
        self.leftkeyNoOrigin=10
        self.path=path
        self.pred=''
        self.shouldgenerate=0
        self.alsoIntersectSet=[]
        self.velocityTable=[]
        self.label=0
        self.combinedFeature=[]
        self.oneOrTwo=0
        self.framelist=[]
        self.framelistYin=[]
        #self.leftframelist=[]
        self.closeframelist=[]
        self.read_index=[1,3,4,5,6,7,8,9]
        self.close_read_index=[100,101,102,103,104,105]
        self.dict={}
        self.hogset=[]

        self.framelist_initial=[]
        self.dict_initial={}
        #self.leftframelist_initial=[]
        #self.leftdict_initial={}
        #self.leftdict={}







        self.traintest=''




        #self.getHogFeature()
        #print self.wordName
    def heightenough(self):
        start=1
        end=1
        lastInter=0
        iniInter=0
        for j in range(len(self.framelist_initial)):
            i=self.framelist_initial[j]
            if start==1 and self.dict_initial[i].ftype=='Intersect':
                iniInter+=1
            else:
                start=0
                break
        if iniInter>5:
            iniInter=0


        for j in range(len(self.framelist_initial)-1,0,-1):
            #print j
            i=self.framelist_initial[j]
            #print i
            #print type(i)
            if i=='22':
                xxx=1
            if end==1 and self.dict_initial[i].ftype=='Intersect':
                lastInter+=1
            else:
                end=0
                break
        if lastInter>5:
            lastInter=0


        for j in range(iniInter,len(self.framelist_initial)-lastInter):
            i=self.framelist_initial[j]
            if i==22:
                xxxx=1
            #print self.dict_initial[i].position[1],self.hip
            if self.dict_initial[i].position[1]<self.hip+50:
                self.dict[i]=self.dict_initial[i]
                self.framelist.append(i)
        '''for i in self.leftframelist_initial:
            if self.leftdict_initial[i].position[1]<self.hip+50:
                self.leftdict[i]=self.leftdict_initial[i]
                self.leftframelist.append(i)'''
        self.dict_initial.clear()
        #self.leftdict_initial.clear()
        del self.dict_initial
        #del self.leftdict_initial
        del self.framelist_initial
        #del self.leftframelist_initial

    def loadData(self):
        print self.path
        filename=glob.glob(self.path+"/"+self.wordName+'*.csv')
        with open(filename[0],'rb') as Label1:
            reader = csv.reader(Label1)
            labelArr1 = []
            last=-999
            #print self.path
            for row in reader:
                if(row[1]!="untracked" and row[1]!="null" and len(row)>100):
                    if last==-999:
                        self.headpos=[int(row[4]),int(row[5])]
                        self.shoulder=float(abs(int(row[11])-int(row[25])))
                        self.hipx=int(row[88])
                        self.hip=int(row[89])
                        self.tall=float(self.hip-self.headpos[1])
                        break

        with open(filename[0],'rb') as Label1:
            reader = csv.reader(Label1)
            for row in reader:
                if row[0]=='5':
                    pass
                touch=0
                position=[]

                if(row[1]=="null" and len(row)>3):
                    labelArr1.append(row)
                    row2=[]
                    self.fr=int(row[0])

                    if len(row)==7:
                        position=[]


                        position.append(int(row[2]))
                        position.append(int(row[3]))
                        position.append(int(row[4]))
                        position.append(int(row[5]))
                    elif len(row)==8:
                        position.append(int(row[6]))
                        position.append(int(row[7]))
                        position.append(int(row[6]))
                        position.append(int(row[7]))

                    #if float(row[3])>self.hip+50:
                    #    continue
                    if row[0]==last:
                        continue
                        #self.dict[row[0]].setPositionInter(position,0)
                row2=[]
                if(row[1]!="untracked" and row[1]!="null" and len(row)>100):
                    labelArr1.append(row)
                    row2=[]
                    self.fr=int(row[0])

                    for index in self.read_index:
                        #print self.path,row[0],row[1],index,len(row2),len(row),row2,row
                        row2.append(float(row[index*7+1]))
                        row2.append(float(row[index*7+2]))
                        row2.append(float(row[index*7+3]))
                    #print row[0],row2[22],row2[25]-0.12
                    '''if row2[22]<row2[25]-0.12:
                        continue'''
                    '''if len(row)==105:
                        if float(row[104])>self.hip+50:
                            continue
                    elif len(row)==104:
                        if float(row[100])>self.hip+50 and float(row[102])>self.hip+50:
                            continue'''
                    if row[0]==last and row[1]!="null":
                        continue
                    if float(row[3])-float(row[66])<0.3:
                        touch=1
                    #self.dict[row[0]]=frame(row[0],row2)
                    #self.framelist.append(row[0])

                    if len(row)>=102:
                        position=[]

                        if len(row)==105:
                            position.append(int(row[103]))
                            position.append(int(row[104]))
                            position.append(int(row[103]))
                            position.append(int(row[104]))
                            frameinter=1
                        elif len(row)==104:
                            frameinter=0
                            position.append(int(row[99]))
                            position.append(int(row[100]))
                            position.append(int(row[101]))
                            position.append(int(row[102]))
                        #assert len(row)==105 or len(row)==104


                        #self.closeframelist.append(row[0])


                if position==[]:
                    continue
                rightimg=None
                leftimg=None
                #print self.path+"/handshape/"+str(row[0])+"*Intersect*_C*.jpg"
                interset=glob.glob(self.path+"/handshape/"+str(row[0])+"_*Intersect*_C*.jpg")
                bothset=glob.glob(self.path+"/handshape/"+str(row[0])+"_*Both*_C*.jpg")
                rightset0=glob.glob(self.path+"/handshape/"+str(row[0])+"_*Right*_C*.jpg")
                leftset0=glob.glob(self.path+"/handshape/"+str(row[0])+"_*Left*_C*.jpg")
                rightset=bothset+rightset0+leftset0
                leftset=glob.glob(self.path+"/handshape/left/"+str(row[0])+"_*_C*.jpg")
                if interset!=[]:
                    ftype='Intersect'
                    #rightimg=cv2.imread(interset[0])
                elif leftset!=[] and rightset!=[]:
                    ftype='Both'
                    leftimg=cv2.imread(leftset[0])
                    #rightimg=cv2.imread(rightset[0])
                elif leftset==[] and rightset!=[]:
                    ftype='Right'
                    #rightimg=cv2.imread(rightset[0])
                #elif leftset!=[] and rightset==[]:
                #    ftype='Left'
                #    leftimg=cv2.imread(leftset[0])
                else:
                    ftype='None'
                self.dict[self.fr]=frame(self.fr,row2,ftype,rightimg,leftimg,position,touch)

                self.framelist.append(self.fr)
                if ftype!='None':
                    self.framelistYin.append(self.fr)



                last=row[0]

        flag=0
        for f in self.framelist:
            if self.dict[f].ftype!='None':
                flag=1
                break
        if flag==0:
            return 0

        if len(self.framelist)<5:
            return 0
        else:
            return 1

    def loadDatadevisign(self):
        filename=glob.glob(self.path+"/"+self.sampleName+'*.csv')
        #print self.path
        with open(filename[0],'rb') as Label1:
            reader = csv.reader(Label1)
            labelArr1 = []
            last=-999
            for row in reader:
                if len(row)<20:
                    continue
                self.headpos=[int(row[4]),int(row[5])]
                self.shoulder=float(abs(int(row[9])-int(row[14])))
                self.leftinix=int(row[18])
                self.leftiniy=int(row[19])
                #print row[16],row[18]
                self.hipx=int((int(row[16])+int(row[18]))/2)
                #self.hip=int((int(row[17])+int(row[19]))/2)
                self.hip=int(row[19])
                self.tall=float(self.hip-self.headpos[1])
                break

        with open(filename[0],'rb') as Label1:
            reader = csv.reader(Label1)
            for row in reader:
                touch=0
                if len(row)>20:

                    labelArr1.append(row)
                    row2=[]
                    self.fr=int(row[0])
                    '''for index in self.read_index:
                        row2.append(float(row[index*7+1]))
                        row2.append(float(row[index*7+2]))
                        row2.append(float(row[index*7+3]))'''


                    if len(row)>17:
                        if float(row[17])>self.hip-15:
                            continue

                    #elif len(row)==21:
                    #    if float(row[100])>self.hip+50 and float(row[102])>self.hip+50:
                    #        continue


                    #if float(row[3])-float(row[66])<0.3:
                    #    touch=1



                    position=[]

                    if len(row)==22:
                        position.append(int(row[20]))
                        position.append(int(row[21]))
                        position.append(int(row[20]))
                        position.append(int(row[21]))
                        frameinter=1
                    elif len(row)==21:
                        frameinter=0
                        position.append(int(row[16]))
                        position.append(int(row[17]))
                        position.append(int(row[18]))
                        position.append(int(row[19]))
                        if abs(int(row[16])-self.leftinix)<5 and abs(int(row[17])-self.leftiniy)<5:
                            continue




                    assert len(row)==22 or len(row)==21


                    #self.closeframelist.append(row[0])


                    if position==[]:
                        continue
                    rightimg=None
                    leftimg=None
                    #print self.path+"/handshape/"+str(row[0])+"*Intersect*_C*.jpg"
                    interset=glob.glob(self.path+"/handshape/"+str(row[0])+"_*Intersect*_C*.jpg")
                    bothset=glob.glob(self.path+"/handshape/"+str(row[0])+"_*Both*_C*.jpg")
                    rightset0=glob.glob(self.path+"/handshape/"+str(row[0])+"_*Right*_C*.jpg")
                    leftset0=glob.glob(self.path+"/handshape/"+str(row[0])+"_*Left*_C*.jpg")
                    rightset=bothset+rightset0+leftset0
                    leftset=glob.glob(self.path+"/handshape/left/"+str(row[0])+"_*_C*.jpg")
                    if interset!=[]:
                        ftype='Intersect'
                        #rightimg=cv2.imread(interset[0])
                    elif leftset!=[] and rightset!=[]:
                        ftype='Both'
                        #leftimg=cv2.imread(leftset[0])
                        #rightimg=cv2.imread(rightset[0])
                    elif leftset==[] and rightset!=[]:
                        ftype='Right'
                        #rightimg=cv2.imread(rightset[0])
                    #elif leftset!=[] and rightset==[]:
                    #    ftype='Left'
                    #    leftimg=cv2.imread(leftset[0])
                    else:
                        ftype='None'
                    self.dict[self.fr]=frame(self.fr,row2,ftype,rightimg,leftimg,position,touch)
                    self.framelist.append(self.fr)
        flag=0
        for f in self.framelist:
            if self.dict[f].ftype!='None':
                flag=1
                break
        if flag==0:
            return 0

        if len(self.framelist)<5:
            return 0
        else:
            return 1
        #self.dict_initial[self.fr]=frame(self.fr,row2,ftype,rightimg,leftimg,position,touch)
        #self.framelist_initial.append(self.fr)
        #last=row[0]

        #return 1



    def consFeature(self):
        stop=0
        laststop=0
        data=[]
        frameno=0
        fast=1
        avlist=[]
        avLen=5
        framel=[]
        distancel=[]
        for f in self.framelist:
            frameno+=1
            data.append(self.dict[f].position)
        #print self.path
        for i in range(len(self.framelist)-1):

            dis=math.sqrt((data[i+1][0]-data[i][0])**2+(data[i+1][1]-data[i][1])**2)
            framel.append(i)
            distancel.append(dis)
            #print i,dis
            if len(avlist)<avLen:
                avlist.append(dis)
            else:
                avlist=avlist[1:]
                avlist.append(dis)
                average=sum(avlist)/avLen
                if average<3 and i-laststop>10 and fast==1:
                    fast=0
                    stop+=1
                    laststop=i
                elif average>3:
                    fast=1
        '''a=plt.plot(framel,distancel)
        plt.savefig(self.path+"/plt/velocity.jpg")
        plt.clf()'''
        '''fstop=open(self.path+"/stop.txt","w")
        fstop.write(str(stop))
        fstop.close()'''





















    def testSignWord(self,datamode):
        files=os.listdir(self.path+"/handshape/")
        if datamode==0:
            f=self.loadData()
        else:
            f=self.loadDatadevisign()


        if(len(files)<5):
            return 0
        elif f==0:
            return 0
        else:
            return 1

    def consTrajectory(self):
        if self.intersect==0:
            pointNo=200
        else:
            pointNo=200

        #print self.path
        if(self.framelist==[] or len(self.framelist)==1 or len(self.framelist)==2):
            return 0
        #print self.framelist
        data=[]
        velocity=[]
        frameno=0
        headx=self.headpos[0]
        heady=self.headpos[1]
        shoulder=self.shoulder+0.000001
        tall=self.tall
        for f in self.framelist:
            #print self.dict[f].data[22],self.dict[f].data[25]-0.12

            if self.dict[f].position==[0,0,0,0]:
                continue
            #if self.dict[f].position[1]>self.hip-10000:
            #    continue
            frameno+=1
            data.append(self.dict[f].position)
            velocity.append(self.dict[f].value2)
        #print self.sampleName
        #self.displacement=hodHand(data,9,2)
        traje100=[]
        self.xtrajectory=[]
        self.ytrajectory=[]
        self.xlefttrajectory=[]
        self.ylefttrajectory=[]
        xtrapos=[]
        ytrapos=[]
        xlefttrapos=[]
        ylefttrapos=[]
        totalDis=0
        biglist=[]
        bigvelocity=[]
        for i in range(frameno-1):
            dis=math.sqrt((data[i+1][0]-data[i][0])**2+(data[i+1][1]-data[i][1])**2)
            if dis>3:
                biglist.append(data[i])
                bigvelocity.append(velocity[i])
                totalDis+=math.sqrt((data[i+1][0]-data[i][0])**2+(data[i+1][1]-data[i][1])**2)
        data=biglist
        velocity=bigvelocity
        if data==[] or len(data)==1:
            return 0
        totalDis=0
        for i in range(len(biglist)-1):
            totalDis+=math.sqrt((data[i+1][0]-data[i][0])**2+(data[i+1][1]-data[i][1])**2)


        step0=totalDis/pointNo

        step=step0
        cuframe=0
        #steped=0
        framed=0
        newframe=1
        velocityTrajectory=[]
        for i in range(pointNo):
            #print "right i",i
            #print len(data)
            #print cuframe
            if cuframe==len(data)-1 or cuframe>len(data)-1:
                while len(self.xtrajectory)<pointNo:
                    self.xtrajectory.append((float(data[len(data)-1][0])-headx)/shoulder)
                    self.ytrajectory.append((data[len(data)-1][1]-heady)/tall)
                    velocityTrajectory.append(velocity[len(data)-1])
                break
            framedis=math.sqrt((data[cuframe+1][0]-data[cuframe][0])**2+(data[cuframe+1][1]-data[cuframe][1])**2)
            while framedis==0:
                cuframe+=1
                if cuframe<len(data)-1:
                    framedis=math.sqrt((data[cuframe+1][0]-data[cuframe][0])**2+(data[cuframe+1][1]-data[cuframe][1])**2)
                else:
                    return 0
            framedis-=framed
            step=step0
            while(step>framedis):
                step-=framedis
                cuframe+=1
                framed=0
                if cuframe<len(data)-1:
                    framedis=math.sqrt((data[cuframe+1][0]-data[cuframe][0])**2+(data[cuframe+1][1]-data[cuframe][1])**2)
                else:
                    self.xtrajectory.append((float(data[len(data)-1][0])-headx)/shoulder)
                    self.ytrajectory.append((data[len(data)-1][1]-heady)/tall)
                    velocityTrajectory.append(velocity[len(data)-1])
                    xtrapos.append(data[len(data)-1][0])
                    ytrapos.append(data[len(data)-1][1])
                    break
            if step<=framedis and cuframe<len(data)-1:
                #print data[cuframe+1]
                #print data[cuframe]


                fraction=(step+framed)/math.sqrt((data[cuframe+1][0]-data[cuframe][0])**2+(data[cuframe+1][1]-data[cuframe][1])**2)
                x=fraction*data[cuframe+1][0]+(1-fraction)*data[cuframe][0]
                y=fraction*data[cuframe+1][1]+(1-fraction)*data[cuframe][1]
                xi=(float(x)-headx)/shoulder
                yi=(float(y)-heady)/tall
                self.xtrajectory.append(xi)
                self.ytrajectory.append(yi)
                velocityTrajectory.append(fraction*velocity[cuframe+1]+(1-fraction)*velocity[cuframe])
                xtrapos.append(x)
                ytrapos.append(y)
                framed+=step
                #print x,y,cuframe

        '''mini=min(self.trajectory)
        for i in range(len(self.trajectory)):
            self.trajectory[i]=self.trajectory[i]-mini'''
        while len(self.xtrajectory)<pointNo:
            self.xtrajectory.append((float(data[len(data)-1][0])-headx)/shoulder)
            self.ytrajectory.append((data[len(data)-1][1]-heady)/tall)
            velocityTrajectory.append(velocity[len(data)-1])

        radius=[]

        minvelocity=min(velocityTrajectory)
        for i in range(len(velocityTrajectory)):
            radius.append(velocityTrajectory[i]-minvelocity)


        m = np.zeros((640,640,3))
        for i in range(0,len(xtrapos)):
            #print int(velocityTrajectory[int(i/2)]*5)
            cv2.circle(m,(int(xtrapos[i]),int(ytrapos[i])),2,(100,100,250),-1)
        cv2.circle(m,(int(self.headpos[0]),int(self.headpos[1])),1,(250,250,250),-1)

        cv2.circle(m,(int(self.headpos[0]),int(self.hip)),1,(250,250,250),-1)
        cv2.imwrite(self.path+"/trajectory.jpg",m)
        leftvelocity=[]
        assert self.xtrajectory!=[]

        #assert len(self.trajectory)==100
        #assert len(radius)==50

        #print self.trajectory
        '''self.trajecctory=normalize_histogram_abs_not1(self.trajectory,20)
        radius=normalize_histogram_abs_not1(radius,10)
        self.trajectory=normalize_histogram(self.trajecctory+radius)'''
        #print self.trajectory


        '''csvfile = file(self.path+'/trajectory.csv', 'wb')
        writer = csv.writer(csvfile)
        writer.writerow(self.trajectory)
        csvfile.close()'''












        if(self.bothseparate==1):

            frameno=0
            data=[]
            for f in self.framelist:
                if self.dict[f].ftype=='Both': #or self.dict[f].ftype=='Left':
                    if self.dict[f].position==[0,0,0,0]:
                        continue
                    frameno+=1
                    data.append(self.dict[f].position)
                    leftvelocity.append(self.dict[f].leftvalue2)

            xtrapos=[]
            ytrapos=[]
            leftbiglist=[]
            leftbigvelocity=[]
            totalDis=0
            for i in range(frameno-1):
                dis=math.sqrt((data[i+1][2]-data[i][2])**2+(data[i+1][3]-data[i][3])**2)
                if dis>3:
                    leftbiglist.append(data[i])
                    leftbigvelocity.append(leftvelocity[i])
                    totalDis+=math.sqrt((data[i+1][2]-data[i][2])**2+(data[i+1][3]-data[i][3])**2)
            data=leftbiglist
            leftvelocity=leftbigvelocity
            if data==[] or len(data)==1:
                return 0
            totalDis=0
            for i in range(len(leftbiglist)-1):
                totalDis+=math.sqrt((data[i+1][2]-data[i][2])**2+(data[i+1][3]-data[i][3])**2)

            step0=totalDis/pointNo
            leftvelocityTra=[]
            step=step0
            cuframe=0
            #steped=0
            framed=0
            newframe=1
            for i in range(pointNo):
                if cuframe==len(data)-1 or cuframe>len(data)-1:
                    while len(self.xlefttrajectory)<pointNo:
                        self.xlefttrajectory.append((float(data[len(data)-1][2])-headx)/shoulder)
                        self.ylefttrajectory.append((data[len(data)-1][3]-heady)/tall)
                        leftvelocityTra.append(leftvelocity[len(data)-1])
                        xlefttrapos.append(data[len(data)-1][2])
                        ylefttrapos.append(data[len(data)-1][3])
                    break
                framedis=math.sqrt((data[cuframe+1][2]-data[cuframe][2])**2+(data[cuframe+1][3]-data[cuframe][3])**2)
                while framedis==0:
                    cuframe+=1
                    if cuframe<len(data)-1:
                        framedis=math.sqrt((data[cuframe+1][2]-data[cuframe][2])**2+(data[cuframe+1][3]-data[cuframe][3])**2)
                    else:
                        while self.xlefttrajectory<pointNo:
                            self.xlefttrajectory.append((float(data[len(data)-1][2])-headx)/shoulder)
                            self.ylefttrajectory.append((data[len(data)-1][3]-heady)/tall)
                            leftvelocityTra.append(leftvelocity[len(data)-1])
                            xlefttrapos.append(data[len(data)-1][2])
                            ylefttrapos.append(data[len(data)-1][3])
                        #return 0
                framedis-=framed
                step=step0
                while(step>framedis):
                    step-=framedis
                    cuframe+=1
                    framed=0
                    if cuframe<len(data)-1:
                            framedis=math.sqrt((data[cuframe+1][2]-data[cuframe][2])**2+(data[cuframe+1][3]-data[cuframe][3])**2)
                    else:
                            break
                if step<=framedis and cuframe<len(data)-1:
                    fraction=(step+framed)/math.sqrt((data[cuframe+1][2]-data[cuframe][2])**2+(data[cuframe+1][3]-data[cuframe][3])**2)
                    x=fraction*data[cuframe+1][2]+(1-fraction)*data[cuframe][2]
                    y=fraction*data[cuframe+1][3]+(1-fraction)*data[cuframe][3]
                    xi=(float(x)-headx)/shoulder
                    yi=(float(y)-heady)/tall
                    self.xlefttrajectory.append(xi)
                    self.ylefttrajectory.append(yi)
                    leftvelocityTra.append(fraction*leftvelocity[cuframe+1]+(1-fraction)*leftvelocity[cuframe])
                    xlefttrapos.append(x)
                    ylefttrapos.append(y)
                    framed+=step
            while len(self.xlefttrajectory)<pointNo:
                self.xlefttrajectory.append((float(data[len(data)-1][2])-headx)/shoulder)
                self.ylefttrajectory.append((data[len(data)-1][3]-heady)/tall)
                leftvelocityTra.append(leftvelocity[len(data)-1])
                xlefttrapos.append(data[len(data)-1][2])
                ylefttrapos.append(data[len(data)-1][3])

            '''mini=min(self.lefttrajectory)
            for i in range(len(self.lefttrajectory)):
                self.lefttrajectory[i]=self.lefttrajectory[i]-mini'''
            radius=[]
            minvelocity=min(leftvelocityTra)
            for i in range(len(leftvelocityTra)):
                radius.append(leftvelocityTra[i]-minvelocity)

            m = np.zeros((500,500,3))

            for i in range(0,len(self.xlefttrajectory)):
                cv2.circle(m,(int(self.headpos[0]),int(self.headpos[1])),1,(250,250,250),-1)
                cv2.circle(m,(int(self.headpos[0]),int(self.hip)),1,(250,250,250),-1)
                cv2.circle(m,(int(xlefttrapos[i]),int(ylefttrapos[i])),2,(100,100,250),-1)
            cv2.imwrite(self.path+"/lefttrajectory.jpg",m)
            #assert len(self.lefttrajectory)==100
            #assert len(radius)==50
            '''self.lefttrajecctory=normalize_histogram_abs_not1(self.lefttrajectory,20)
            radius=normalize_histogram_abs_not1(radius,10)
            self.lefttrajectory=self.lefttrajecctory+radius'''

            '''csvfile = file(self.path+'/trajectory.csv', 'wb')
            writer = csv.writer(csvfile)
            writer.writerow(self.trajectory+self.lefttrajectory)
            csvfile.close()'''




    def getVelo(self):
        first=1
        for f in self.framelist:


            frame=self.dict[f]

            if frame.position==[]:
                continue
            if first==0:


                x=frame.position[0]
                y=frame.position[1]

                velocity=math.sqrt((x-px)**2+(y-py)**2)
                frame.height=frame.position[1]
                frame.velocity=velocity
                px=x
                py=y

            else:
                #print frame.data
                px=frame.position[0]
                py=frame.position[1]
                first=0
        try:
            self.dict[self.framelist[0]].velocity=self.dict[self.framelist[1]].velocity
        except:
            pass

        #fastest
        '''fastest=0
        i=0
        while(1):
            f=self.framelist[i]
            frame=self.dict[f]
            if frame.ftype!='None' or i==len(self.framelist)-1:
                break

            i+=1
            if(frame.velocity>fastest):
                fastest=frame.velocity


        stop1=i

        i=len(self.framelist)-1
        while(1):
            f=self.framelist[i]
            frame=self.dict[f]
            if frame.ftype!='None' or i==0:
                break

            i-=1
            if(frame.velocity>fastest):
                fastest=frame.velocity

        stop2=i
        #forward
        i=0
        while(1):
            f=self.framelist[i]
            frame=self.dict[f]
            if i==stop1:
                break
            i+=1

            frame.velocity=fastest

        #back forward
        i=len(self.framelist)-1
        while(1):
            f=self.framelist[i]
            frame=self.dict[f]
            if i==stop2:
                break
            i-=1

            frame.velocity=fastest
        if fastest>0:
            for i in range(stop1,stop2):
                f=self.framelist[i]
                frame=self.dict[f]
                if frame.velocity>fastest:
                    frame.velocity=fastest'''


        #left
        first=1
        for f in self.framelist:
            frame=self.dict[f]
            if frame.data==[]:
                continue
            if first==0:


                x=frame.position[2]
                y=frame.position[3]
                velocity=math.sqrt((x-px)**2+(y-py)**2)
                frame.leftheight=frame.position[3]
                frame.leftvelocity=velocity
                px=x
                py=y

            else:
                #print frame.data
                px=frame.position[2]
                py=frame.position[3]
                first=0
        try:
            self.dict[self.framelist[0]].leftvelocity=self.dict[self.framelist[1]].leftvelocity
        except:
            pass




        #fastest
        '''fastest=0
        i=0
        while(1):
            f=self.framelist[i]
            frame=self.dict[f]
            if frame.ftype!='None' or i==len(self.framelist)-1:
                break

            i+=1
            if(frame.leftvelocity>fastest):
                fastest=frame.leftvelocity


        stop1=i

        i=len(self.framelist)-1
        while(1):
            f=self.framelist[i]
            frame=self.dict[f]
            if frame.ftype!='None' or i==0:
                break

            i-=1
            if(frame.leftvelocity>fastest):
                fastest=frame.leftvelocity

        stop2=i
        #forward
        i=0
        while(1):
            f=self.framelist[i]
            frame=self.dict[f]
            if i==stop1:
                break
            i+=1

            frame.leftvelocity=fastest

        #back forward
        i=len(self.framelist)-1
        while(1):
            f=self.framelist[i]
            frame=self.dict[f]
            if i==stop2:
                break
            i-=1

            frame.leftvelocity=fastest
        if fastest>0:
            for i in range(stop1,stop2):
                f=self.framelist[i]
                frame=self.dict[f]
                if frame.leftvelocity>fastest:
                    frame.leftvelocity=fastest'''

        '''v=[]
        for f in self.framelist:
            frame=self.dict[f]
            if frame.data==[]:
                continue
            vh=[frame.num,frame.velocity,frame.height,frame.leftvelocity,frame.leftheight]
            v.append(vh)

        self.velocityTable=v[1:]'''


        '''first=1
        for f in self.framelist:
        .    frame=self.dict[f]
            if first==0:
                x=frame.position[0]
                y=frame.position[1]
                velocity=math.sqrt((x-px)**2+(y-py)**2)
                frame.height=frame.position[1]
                frame.velocity=velocity
                px=x
                py=y
            else:
                #print frame.data
                px=frame.position[0]
                py=frame.position[1]
                first=0
        first=1
        for f in self.framelist:
            frame=self.dict[f]
            if first==0:


                x=frame.position[2]
                y=frame.position[3]
                velocity=math.sqrt((x-px)**2+(y-py)**2)
                frame.leftheight=frame.position[3]
                frame.leftvelocity=velocity
                px=x
                py=y

            else:
                #print frame.data
                px=frame.position[2]
                py=frame.position[3]
                first=0

        v=[]
        for f in self.framelist:
            frame=self.dict[f]
            vh=[frame.num,frame.velocity,frame.height,frame.leftvelocity,frame.leftheight]
            v.append(vh)
        self.velocityTable=v[1:]'''
























    def getInter(self):
        print self.path
        inter=0
        for f in self.framelist:
            if self.dict[f].ftype=='Intersect':
                inter+=1


        if self.bothseparate==1:

            if inter>0:
                self.intersect=1
                self.bothseparate=0
                self.leftkeyNo=0

            else:
                self.intersect=0
                self.leftkeyNo=self.leftkeyNoOrigin
        else:
            if inter>10:
                self.intersect=1
                self.bothseparate=0
                self.leftkeyNo=0
        #print "inter",inter,self.intersect



    def getBothSeparate(self):
        leftnum=0
        for f in self.framelist:
            if self.dict[f].ftype=='Both':# or self.dict[f].ftype=='Left':
                leftnum+=1
        if leftnum>2:
            self.bothseparate=1
            self.leftkeyNo=self.leftkeyNoOrigin
        else:
            self.bothseparate=0
            self.leftkeyNo=0








    '''def createIntersectImage(self,f):

        leftimgs=glob.glob(self.path+"/handshape/left/"+f+"_*_C*.jpg")
        rightimgs=glob.glob(self.path+"/handshape/"+f+"_*_C*.jpg")

        if leftimgs!=[] and rightimgs!=[]:
            leftimg=cv2.imread(leftimgs[0])
            rightimg=cv2.imread(rightimgs[0])

            leftshape=leftimg.shape
            rightshape=rightimg.shape

            back=numpy.zeros((800,800,3),numpy.uint8)
            position=self.dict[f].position
            r1=rightshape[0]
            r2=rightshape[1]
            l1=leftshape[0]
            l2=leftshape[1]


            if rightshape[0]%2==1:
                edge=numpy.zeros((1,rightshape[1],3),numpy.uint8)
                rightimg=numpy.vstack((rightimg,edge))
                r1+=1
            if rightshape[1]%2==1:
                edge=numpy.zeros((r1,1,3),numpy.uint8)
                rightimg=numpy.hstack((rightimg,edge))
                r2+=1
            if leftshape[0]%2==1:
                edge=numpy.zeros((1,leftshape[1],3),numpy.uint8)
                leftimg=numpy.vstack((leftimg,edge))
                l1+=1
            if leftshape[1]%2==1:
                edge=numpy.zeros((l1,1,3),numpy.uint8)
                leftimg=numpy.hstack((leftimg,edge))
                l2+=1
            back[(position[3]-l1/2):(position[3]+l1/2),position[2]-l2/2:position[2]+l2/2,0:3]=leftimg
            back[(position[1]-r1/2):(position[1]+r1/2),(position[0]-r2/2):(position[0]+r2/2),0:3]=rightimg
            output=back[min((position[3]-l1/2),(position[1]-r1/2)):max((position[3]+l1/2),(position[1]+r1/2)),min(position[2]-l2/2,position[0]-r2/2):max(position[2]+l2/2,position[0]+r2/2)]
            if os.path.exists(self.path+"/handshape/generated/")==0:
                os.mkdir(self.path+"/handshape/generated/")
            cv2.imwrite(self.path+"/handshape/generated/"+f+".jpg",output)
            self.alsoIntersectSet.append(output)'''




    def getVelocity(self):
        print self.path
        self.indexList=[]
        self.leftindexList=[]
        for f in self.framelist:
            self.dict[f].value=-10*float(self.dict[f].velocity)-0.05*abs(self.dict[f].num-len(self.framelist)/2)
            self.dict[f].leftvalue=-10*float(self.dict[f].leftvelocity)-0.05*abs(self.dict[f].num-len(self.framelist)/2)
            self.dict[f].value2=float(-self.dict[f].position[1])-10*float(self.dict[f].velocity)-0.005*abs(self.dict[f].num-len(self.framelist)/2)
            self.dict[f].leftvalue2=float(-self.dict[f].position[3])-10*float(self.dict[f].leftvelocity)-0.005*abs(self.dict[f].num-len(self.framelist)/2)

        if self.bothseparate==0 and self.intersect==0:
            for f in self.framelist:
                if self.dict[f].ftype=='Both' or self.dict[f].ftype=='Right':
                    self.indexList.append(self.dict[f].num)
        elif self.bothseparate==1:
            for f in self.framelist:
                if self.dict[f].ftype=='Both' or self.dict[f].ftype=='Right':
                    self.indexList.append(self.dict[f].num)
                if self.dict[f].ftype=='Both':
                    self.leftindexList.append(self.dict[f].num)
            if self.leftindexList==[]:
                for frameno in self.framelist:
                    self.indexList.append(frameno)
            if(len(self.leftindexList)<self.leftkeyNo):
                self.leftkeyNo=len(self.leftindexList)

        elif self.intersect==1:
            for f in self.framelist:
                if self.dict[f].ftype=='Intersect' or self.dict[f].ftype=='Both':
                    self.indexList.append(self.dict[f].num)

        if self.indexList==[]:
            for frameno in self.framelist:
                if self.dict[frameno].ftype=='None':
                    continue
                self.indexList.append(frameno)



        if(len(self.indexList)<self.keyNo):
            self.keyNo=len(self.indexList)



    def modifyKeySign(self):
        #print self.path
        formerimp=glob.glob(self.path+"/handshape/"+"*_*_C"+"#.jpg")
        self.handshapes=[]
        for imp in formerimp:
            os.rename(imp,imp[:-5]+".jpg")
        for i in range(len(self.topIndex)):
            #print self.topIndex,i
            newimp=glob.glob(self.path+"/handshape/"+str(self.topIndex[i])+"_*_C*"+".jpg")
            self.handshapes.append(cv2.resize(cv2.imread(newimp[0]),(227,227)))
            if newimp==[]:
                continue
            oldname=newimp[0]
            os.rename(oldname,oldname[:-4]+"#.jpg")

        if self.bothseparate==1:
            formerimp=glob.glob(self.path+"/handshape/left/"+"*_*_C"+"#.jpg")
            for imp in formerimp:
                os.rename(imp,imp[:-5]+".jpg")

            for i in range(len(self.lefttopIndex)):

                newimp=glob.glob(self.path+"/handshape/left/"+str(self.lefttopIndex[i])+"_*_C"+".jpg")

                oldname=newimp[0]
                os.rename(oldname,oldname[:-4]+"#.jpg")
    def modifyKeySign2(self):
        f=open(self.path+"/tmpkey.txt","w")
        for i in range(len(self.topIndex)):
            f.write(str(self.topIndex[i])+' ')
        f.close()

    def findTopPos(self):
        self.topPosition=[]

        for f in self.topIndex:
            frame=self.dict[f]
            position=frame.position
            xru=(position[0]-self.headpos[0])/self.shoulder
            yru=(position[1]-self.headpos[1])/self.tall
            xlu=(position[2]-self.headpos[0])/self.shoulder
            ylu=(position[3]-self.headpos[1])/self.tall

            xrd=(position[0]-self.hipx)/self.shoulder
            yrd=(position[1]-self.hip)/self.tall
            xld=(position[2]-self.hipx)/self.shoulder
            yld=(position[3]-self.hip)/self.tall

            topposi=[xru,yru,xlu,ylu,xrd,yrd,xld,yld]

            self.topPosition.append(topposi)



    def findTopHandshape(self):
        print self.path
        self.top=[]
        self.topIndex=[]
        #print self.value
        for i in range(self.keyNo):
            self.top.append(self.dict[self.indexList[i]].value)
            self.topIndex.append(self.indexList[i])
        #top5Index=[indexList[0],indexList[1],indexList[2],indexList[3],indexList[4]]
        for i in range(self.keyNo,len(self.indexList)):
            if(self.dict[self.indexList[i]].value>min(self.top)):
                if self.indexList[i] in self.topIndex:
                    continue
                ind=self.top.index(min(self.top))

                self.top[ind]=self.dict[self.indexList[i]].value
                self.topIndex[ind]=self.indexList[i]
        flag=0
        inter=0
        single=0
        if self.intersect==1:
            '''for t in self.topIndex:
                if self.dict[t].ftype=='Intersect':
                    inter+=1
                else:
                    single+=1
            if inter<single:
                intersect=[]
                for i in self.framelist:
                    if self.dict[i].ftype=='Intersect':
                        intersect.append(i)
                num=0
                for t in range(len(self.topIndex)):
                    f=self.topIndex[t]
                    if self.dict[f].ftype!='Intersect':
                        if num>=len(intersect):
                            num=len(intersect)-1
                        self.topIndex[t]=intersect[num]
                        inter+=1
                        single-=1
                        num+=1
                    if inter>=single:
                        break'''


            for t in self.topIndex:
                if self.dict[t].ftype=='Intersect':
                    flag=1
                    break
            if flag==0:
                for f in self.framelist:
                    if self.dict[f].ftype=='Intersect':
                        self.keyNo+=1
                        self.topIndex.append(f)
                        break
        self.topIndex.sort()

        self.singlekeyNo=0
        self.interkeyNo=0

        for i in range(len(self.topIndex)):
            if self.dict[self.topIndex[i]].ftype!='Intersect':
                self.singlekeyNo+=1
            else:
                self.interkeyNo+=1












        if self.bothseparate==1:
            self.lefttop=[]
            self.lefttopIndex=[]

            for i in range(self.leftkeyNo):
                self.lefttop.append(self.dict[self.leftindexList[i]].leftvalue)
                self.lefttopIndex.append(self.leftindexList[i])

            for i in range(self.leftkeyNo,len(self.leftindexList)):
                if(self.dict[self.leftindexList[i]].leftvalue>min(self.lefttop)):
                    if self.leftindexList[i] in self.lefttopIndex:
                        continue
                    ind=self.lefttop.index(min(self.lefttop))

                    self.lefttop[ind]=self.dict[self.leftindexList[i]].leftvalue
                    self.lefttopIndex[ind]=self.leftindexList[i]
            self.lefttopIndex.sort()














    def getHogFeature(self):
        print self.path
        if self.intersect==0 and self.bothseparate==0:
            hogSet=[]

            for f in self.topIndex:
            #for i in range(len(self.framelist)):

                frame=self.dict[f]
                if len(glob.glob(self.path+"/handshape/"+str(f)+"_*_C*.jpg"))==0:
                    continue
                image=cv2.imread(glob.glob(self.path+"/handshape/"+str(f)+"_*_C*.jpg")[0])

                if image!=None:
                    sp=image.shape
                    if sp[0]>sp[1]:
                        img2=cv2.copyMakeBorder(image, 0,0, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2), cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
                    else:
                        img2=cv2.copyMakeBorder(image, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2),0,0, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
                    img3=cv2.resize(img2,(128,128))
                    image=img3/255.0
                    image = color.rgb2gray(image)
                    image = color.rgb2gray(image)
                    fd, hog_image = hog(image, orientations=9, pixels_per_cell=(32,32),cells_per_block=(2, 2), visualise=True)
                    hogSet.append(fd)
            self.hogFeature=hogmodule.findKey(hogSet)
        if self.bothseparate==1:
            hogSet=[]
            for f in self.topIndex:
            #for i in range(len(self.framelist)):
                frame=self.dict[f]
                if len(glob.glob(self.path+"/handshape/"+str(f)+"_*_C*.jpg"))==0:
                    continue
                image=cv2.imread(glob.glob(self.path+"/handshape/"+str(f)+"_*_C*.jpg")[0])
                if image!=None:
                    sp=image.shape
                    if sp[0]>sp[1]:
                        img2=cv2.copyMakeBorder(image, 0,0, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2), cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
                    else:
                        img2=cv2.copyMakeBorder(image, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2),0,0, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
                    img3=cv2.resize(img2,(128,128))
                    image=img3/255.0
                    image = color.rgb2gray(image)
                    fd, hog_image = hog(image, orientations=9, pixels_per_cell=(60,60),cells_per_block=(2, 2), visualise=True)
                    hogSet.append(fd)

            self.hogFeature=hogmodule.findKey(hogSet)
        if self.intersect==1:
            hogSet=[]
            for f in self.topIndex:
            #for i in range(len(self.framelist)):

                frame=self.dict[f]
                if len(glob.glob(self.path+"/handshape/"+str(f)+"_*_C*.jpg"))==0:
                    continue
                image=cv2.imread(glob.glob(self.path+"/handshape/"+str(f)+"_*_C*.jpg")[0])
                if image!=None and frame.ftype=='Intersect':
                    sp=image.shape
                    if sp[0]>sp[1]:
                        img2=cv2.copyMakeBorder(image, 0,0, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2), cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
                    else:
                        img2=cv2.copyMakeBorder(image, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2),0,0, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
                    img3=cv2.resize(img2,(128,128))
                    image=img3/255.0
                    image = color.rgb2gray(image)
                    image = color.rgb2gray(image)
                    fd, hog_image = hog(image, orientations=9, pixels_per_cell=(60,60),cells_per_block=(2, 2), visualise=True)
                    hogSet.append(fd)
            self.hogFeature=hogmodule.findKey(hogSet)

        '''elif self.intersect==1 and self.shouldgenerate==1:
            hogSet=[]
            for img in self.alsoIntersectSet:
                sp=img.shape
                img2=cv2.copyMakeBorder(img, 0,0, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2), cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
                img3=cv2.resize(img2,(128,128))
                image=img3/255.0
                image = color.rgb2gray(image)
                fd, hog_image = hog(image, orientations=9, pixels_per_cell=(32,32),cells_per_block=(2, 2), visualise=True)
                hogSet.append(fd)'''



    def idvdCaffeFeatureInter(self,img_sum,featureTotal):
        print self.path
        feature=[]

        for i in range(self.keyNo):
            # print img_sum,i
            feat = featureTotal[img_sum+i]
            #feat2 = featureTotal2[img_sum+i]
            #print feat.index(max(feat))
            feature.append(feat)
            #feature2.append(feat2)
        #self.handshapes=feature
        self.handshape=self.pooling(feature,1)


    def idvdCaffeFeature(self,img_sum,img_sum_inter,featureTotal,featureTotal2):
        #print self.path
        assert self.singlekeyNo+self.interkeyNo==self.keyNo
        feature=[]
        feature2=[]
        leftfeature=[]
        leftfeature2=[]
        for i in range(self.singlekeyNo):
            feat = featureTotal[img_sum+i]
            feature.append(feat)
        img_sum+=self.singlekeyNo
        for i in range(self.interkeyNo):
            feat = featureTotal2[img_sum_inter+i]
            feature.append(feat)
        img_sum_inter+=self.interkeyNo
        for i in range(self.leftkeyNo):
            leftfeat = featureTotal[img_sum+i]
            leftfeature.append(leftfeat)
        img_sum+=self.leftkeyNo
        self.handshapefeature=feature
        self.handshape=self.pooling(feature,1)

        if self.bothseparate==1:
            self.lefthandshape=self.pooling(leftfeature,1)
        if self.bothseparate==0 and self.intersect==0:
            self.single2=1

        '''print self.path
        if self.bothseparate==0 and self.intersect==0:
            feature=[]
            for i in range(self.keyNo):
                # print img_sum,i
                feat = featureTotal[img_sum+i]
                #feat2 = featureTotal2[img_sum+i]
                #print feat.index(max(feat))
                feature.append(feat)
                #feature2.append(feat2)
            img_sum+=self.keyNo
            featurenp=np.array(feature)
            featurenp = np.float32(featurenp)
            # define criteria and apply kmeans()
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.011)
            firstset=[]
            secondset=[]
            firstsum=0
            secondsum=0
            firstcnt=0
            secondcnt=0
            self.single2=0
            if len(featurenp)==1:
                firstset.append(feature[0])
                secondset.append(feature[0])

            else:
                ret,belong,center = cv2.kmeans(featurenp,2,criteria,1000,cv2.KMEANS_RANDOM_CENTERS)


                for i in range(len(belong)):
                    if belong[i]==0:
                        firstset.append(feature[i])
                        firstcnt+=1

                        firstsum+=int(self.topIndex[i])
                    else:
                        secondset.append(feature[i])
                        secondcnt+=1

                        secondsum+=int(self.topIndex[i])
                diff=0

                for i in range(1,len(belong)):
                    if belong[i]!=belong[i-1]:
                        diff+=1
                print self.path,diff
                touchNo=0
                for i in self.topIndex:
                    if self.dict[i].touch==1:
                        touchNo+=1

                if diff<2 and touchNo<2 and firstcnt>1 and secondcnt>1:


                    if firstsum/len(firstset) > secondsum/len(secondset):
                        tmp=firstset
                        firstset=secondset
                        secondset=tmp


                    self.single2=1
                else:
                    firstset=feature
                    self.single2=0



            if self.single2==1:
                self.handshape=self.pooling(firstset,1)
                self.handshape2=self.pooling(secondset,1)
            else:
                #self.single2=1
                self.handshape=self.pooling(firstset,1)
                self.handshape2=self.pooling(firstset,1)







        if self.bothseparate==1:
            feature=[]
            for i in range(self.keyNo):
                # print img_sum,i
                feat = featureTotal[img_sum+i]
                #feat2 = featureTotal2[img_sum+i]
                #print feat.index(max(feat))
                feature.append(feat)
                #feature2.append(feat2)
            img_sum+=self.keyNo
            featurenp=np.array(feature)
            featurenp = np.float32(featurenp)
            # define criteria and apply kmeans()
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            firstset=[]
            secondset=[]
            firstsum=0
            secondsum=0
            if len(featurenp)==1:
                firstset.append(feature[0])
                secondset.append(feature[0])
            else:
                ret,belong,center = cv2.kmeans(featurenp,2,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
                for i in range(len(belong)):
                    if belong[i]==0:
                        firstset.append(feature[i])
                    else:
                        secondset.append(feature[i])
                if(len(firstset)<len(secondset)):
                    reverse=1
                    firstset=secondset
                else:
                    reverse=0
                topindextmp=[]
                if reverse==0:
                    for i in range(len(belong)):
                        if belong[i]==0:
                            topindextmp.append(self.topIndex[i])
                else:
                    for i in range(len(belong)):
                        if belong[i]==1:
                            topindextmp.append(self.topIndex[i])
                self.topIndex=topindextmp
            self.handshape=self.pooling(firstset,1)



            leftfeature=[]
            for i in range(self.leftkeyNo):
                # print img_sum,i
                leftfeat = featureTotal[img_sum+i]

                #print feat.index(max(feat))
                leftfeature.append(leftfeat)
            featurenp=np.array(leftfeature)
            featurenp = np.float32(featurenp)
            # define criteria and apply kmeans()
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            firstset=[]
            secondset=[]
            firstsum=0
            secondsum=0
            if len(featurenp)==1:
                firstset.append(leftfeature[0])
                secondset.append(leftfeature[0])
            else:
                ret,belong,center = cv2.kmeans(featurenp,2,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
                for i in range(len(belong)):
                    if belong[i]==0:
                        firstset.append(leftfeature[i])
                    else:
                        secondset.append(leftfeature[i])
                if(len(firstset)<len(secondset)):
                    reverse=1
                    firstset=secondset
                else:
                    reverse=0
                topindextmp=[]
                if reverse==0:
                    for i in range(len(belong)):
                        if belong[i]==0:
                            topindextmp.append(self.lefttopIndex[i])
                else:
                    for i in range(len(belong)):
                        if belong[i]==1:
                            topindextmp.append(self.lefttopIndex[i])
                self.lefttopIndex=topindextmp
            self.lefthandshape=self.pooling(firstset,1)

        if self.intersect==1:
            feature=[]
            for i in range(self.keyNo):
                # print img_sum,i
                feat = featureTotal[img_sum+i]
                feature.append(feat)
            img_sum+=self.keyNo
            self.handshape=self.pooling(feature,1)'''





    def pooling(self,feature,types):

        handshape=[]
        if(types==0):
            #max pooling
            for i in range(len(feature[0])):
                maxvalue=feature[0][i]
                for j in range(len(feature)):
                    if(feature[j][i]>maxvalue):
                        maxvalue=feature[j][i]
                handshape.append(maxvalue)
            return handshape
        if(types==1):
            for i in range(len(feature[0])):
                sum0=0

                for j in range(len(feature)):

                    sum0=sum0+feature[j][i]
                ave=sum0/len(feature)
                handshape.append(ave)
            return handshape
        if(types==2):
            for i in range(len(feature[0])):
                seq=[]
                for j in range(len(feature)):
                    seq.append(feature[j][i])
                seq1=sorted(seq)
                midValue=seq1[len(seq1) // 2]
                handshape.append(midValue)
            return handshape





    def getVariance(self,feature):
        self.hand_index_list=[]
        for x in range(len(feature)):
            hand_index=feature[x].index(max(feature[x]))
            self.hand_index_list.append(hand_index)
        #hand_result.write(str(l)+" "+index2name[l]+" "+str(hand_index_list)+"\n")
        hand_exist=[]
        #print hand_index_list
        self.variance=0
        for x in range(self.keyNo):

            if((self.hand_index_list[x] in hand_exist)==0):
                hand_exist.append(self.hand_index_list[x])
                self.variance+=1







    def loadfeature(self,w1,w2,w3):
        if self.bothseparate==0 and self.intersect==0:
            with open(self.path+'/feature.csv','rb') as Label1:
                reader = csv.reader(Label1)
                cnt=0
                for line in reader:
                    if cnt==0:
                        self.handshape=line
                    if cnt==1:
                        self.hogFeature=line
                    if cnt==2:
                        self.xtrajectory=line
                    if cnt==3:
                        self.ytrajectory=line
                    cnt+=1

            self.combinedFeature=normalize_histogram(self.handshape,w1)+normalize_histogram(self.hogFeature,w2)+normalize_histogram_abs(self.xtrajectory+self.ytrajectory,w3)

        elif self.bothseparate==1:
            with open(self.path+'/feature.csv','rb') as Label1:
                reader = csv.reader(Label1)
                cnt=0
                for line in reader:
                    if cnt==0:
                        self.handshape=line
                    if cnt==1:
                        self.lefthandshape=line
                    if cnt==2:
                        self.hogFeature=line
                    if cnt==3:
                        self.xtrajectory=line
                    if cnt==4:
                        self.ytrajectory=line
                    if cnt==5:
                        self.xlefttrajectory=line
                    if cnt==6:
                        self.ylefttrajectory=line
                    cnt+=1


            self.combinedFeature=normalize_histogram(self.handshape,w1)+normalize_histogram(self.hogFeature,w2)+normalize_histogram(self.lefthandshape,w1)+normalize_histogram_abs(self.xtrajectory+self.ytrajectory,w3)+normalize_histogram_abs(self.xlefttrajectory+self.ylefttrajectory,w3)

        elif self.intersect==1:
            with open(self.path+'/feature.csv','rb') as Label1:
                reader = csv.reader(Label1)

                cnt=0
                for line in reader:
                    if cnt==0:
                        self.handshape=line
                    if cnt==1:
                        self.hogFeature=line
                    if cnt==2:
                        self.xtrajectory=line
                    if cnt==3:
                        self.ytrajectory=line
                    cnt+=1
            self.combinedFeature=normalize_histogram(self.handshape,w1)+normalize_histogram(self.hogFeature,w2)+normalize_histogram_abs(self.xtrajectory+self.ytrajectory,w3)#A+normalize_histogram_abs(self.lefttrajectory)



    def combineFeature(self,w1,w2,w3):
        print self.path

        if self.single2==1:
            feature=[]
            #self.combinedFeature=normalize_histogram_abs(self.hogFeature)
            self.combinedFeature=normalize_histogram(self.handshape,w1)+normalize_histogram(self.hogFeature,w2)+normalize_histogram_abs(self.xtrajectory+self.ytrajectory,w3)
            csvfile = file(self.path+'/featuretry.csv', 'wb')
            writer = csv.writer(csvfile)
            writer.writerow(self.handshape)
            writer.writerow(self.hogFeature)
            writer.writerow(self.xtrajectory)
            writer.writerow(self.ytrajectory)
            csvfile.close()
        elif self.bothseparate==1:

            self.combinedFeature=normalize_histogram(self.handshape,w1)+normalize_histogram(self.hogFeature,w2)+normalize_histogram(self.lefthandshape,w1)+normalize_histogram_abs(self.xtrajectory+self.ytrajectory,w3)+normalize_histogram_abs(self.xlefttrajectory+self.ylefttrajectory,w3)
            csvfile = file(self.path+'/featuretry.csv', 'wb')
            writer = csv.writer(csvfile)
            writer.writerow(self.handshape)
            writer.writerow(self.lefthandshape)
            writer.writerow(self.hogFeature)
            writer.writerow(self.xtrajectory)
            writer.writerow(self.ytrajectory)
            writer.writerow(self.xlefttrajectory)
            writer.writerow(self.ylefttrajectory)
            #self.combinedFeature=normalize_histogram_abs(self.xtrajectory+self.ytrajectory)
            #self.combinedFeature=normalize_histogram(self.handshape)+normalize_histogram(self.lefthandshape)
        elif self.intersect==1:
            self.combinedFeature=normalize_histogram(self.handshape,w1)+normalize_histogram(self.hogFeature,w2)+normalize_histogram_abs(self.xtrajectory+self.ytrajectory,w3)#A+normalize_histogram_abs(self.lefttrajectory)
            csvfile = file(self.path+'/featuretry.csv', 'wb')
            writer = csv.writer(csvfile)
            writer.writerow(self.handshape)
            writer.writerow(self.hogFeature)
            writer.writerow(self.xtrajectory)
            writer.writerow(self.ytrajectory)
            #self.combinedFeature=normalize_histogram_abs(self.xtrajectory+self.ytrajectory)
            #self.combinedFeature=normalize_histogram(self.trajectory)+normalize_histogram(self.lefttrajectory)+normalize_histogram(self.handshape)+normalize_histogram(self.handshape)
            #self.combinedFeature=normalize_histogram(self.displacement)+normalize_histogram(self.trajectory)#+normalize_histogram(self.handshape)#+normalize_histogram(self.hogFeature)
        self.dict.clear()
        del self.dict
        #self.leftdict.clear()
        #del self.leftdict
        self.alsoIntersectSet=[]
        #self.handshape=[]
        #self.displacement=[]
        #self.lefthandshape=[]
        #self.trajectory=[]


    def getdiffi(self,edudic):
        diffs=[]
        if edudic.has_key(self.wordName):
            for i in range(len(edudic[self.wordName])):
                h1=edudic[self.wordName][i]['handshape']
                mini=99999
                for j in range(len(self.topIndex)):
                    h0=self.handshapefeature[j]
                    dis=spatial.distance.cosine(h0,h1)
                    if mini>dis:
                        mini=dis
                diffs.append(mini)

        return diffs
    def enlarge(self,f):
        images=glob.glob(self.path+'/handshape/*#.jpg')

        '''for p in images:
            img=cv2.imread(p)
            sp=img.shape
            if sp[0]>sp[1]:
                img2=cv2.copyMakeBorder(img, 0,0, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2), cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
            else:
                img2=cv2.copyMakeBorder(img, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2),0,0, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
            img3=cv2.resize(img2,(128,128))
            cv2.imwrite(p,img3)'''
        for p in images:
            shutil.copy(p,'/media/lzz/65c50da0-a3a2-4117-8a72-7b37fd81b574/sign/handshapes/hand/handshape/'+self.signer+'+'+self.wordName+'+'+p.split('/')[-1])
            f.write('/media/lzz/65c50da0-a3a2-4117-8a72-7b37fd81b574/sign/handshapes/hand/handshape/'+self.signer+'+'+self.wordName+'+'+p.split('/')[-1]+' '+str(self.label)+'\n')
        #images=glob.glob(self.path+'/handshape/*#.jpg')
        #for p in images:
        #    f.write(p+' '+str(self.label)+'\n')
    def savehdf5(self,f):

        print self.path
        datas=[[[]]]
        labels=[]
        if self.bothseparate==0 and self.bothseparate==0:
            datas=np.zeros((1,400,1,1))
            datas[0,:,0,0]=np.array(self.xtrajectory+self.ytrajectory)
        elif self.bothseparate==1:
            datas=np.zeros((1,800,1,1))
            datas[0,:,0,0]=np.array(self.xtrajectory+self.ytrajectory+self.xlefttrajectory+self.ylefttrajectory)
        elif self.intersect==1:
            datas=np.zeros((1,800,1,1))
            datas[0,:,0,0]=np.array(self.xtrajectory+self.ytrajectory)
        labels=np.zeros((1))

        labels[0]=self.label





        datas=np.array(datas).astype(np.float32)*1000.0

        labels=np.array(labels).astype(np.float32)#.transpose()



        images=np.zeros((1,30,227,227))

        for i in range(10):
            t=min(i,len(self.handshapes)-1)
            print self.handshapes[t].shape
            images[0,i*3,:,:]=self.handshapes[t][:,:,0]
            images[0,i*3+1,:,:]=self.handshapes[t][:,:,0]
            images[0,i*3+2,:,:]=self.handshapes[t][:,:,0]
            #images[0,i*3:i*3+3,:,:]=self.handshapes[t].reshape(3,227,227)

        images=images.astype(np.float32)
        cv2.imwrite('/home/lzz/x.jpg',images[0,0,:,:])


        with h5py.File('/media/lzz/65c50da0-a3a2-4117-8a72-7b37fd81b574/sign/proto_hdf5/hdf5/total/'+self.sampleName.replace(' ','+')+'.h5', 'w') as hdf5file:
            hdf5file['data'] = datas
            hdf5file['image']=  images
            hdf5file['label'] = labels
        f.write('/media/lzz/65c50da0-a3a2-4117-8a72-7b37fd81b574/sign/proto_hdf5/hdf5/total/'+self.sampleName.replace(' ','+')+'.h5\n')