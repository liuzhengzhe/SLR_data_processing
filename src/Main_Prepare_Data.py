'''

@author: qxj
'''
from svmutil import *
from svmmodule import *
import matplotlib
import sqlite3
import math
import numpy
import struct
#from hmm.continuous.GMHMM import GMHMM
#from cluster import KMeansClustering
import time
import random
import matplotlib.pyplot as plt
import marshal, pickle
import svmmodule
from hmmmodule import *
from basic import *
from load import *
from constant_numbers import *
from hodmodule import *
from hog_template import *
from svmmodule import *
import os
#from Tkinter import *
import numpy as np
import cv2
import csv


label2classNo={}

def load_data_no_mog(filelist):
    '''files = os.listdir(path)
    for f in files:
        if os.path.isdir(path+f)==0:
            continue'''
    current_data_index=-1
    data=[]
    label=[]
    namelist=[]
    read_index=[1,3,4,5,6,7,8,9]
    data_index=-1
    namepool={}
    word=-1
    #untrack=[]
    for path in filelist:
        current_data_index+=1
        loc=path.rfind("/")
        name=path[loc:]
        Label1=open(path+"/"+name+'.csv','rb')
        reader = csv.reader(Label1)
        labelArr1 = []
        #untrack.append(0)
        data.append([])
        data_index+=1
        word_name=name[0:name.find(" ")]
        if not namepool.has_key(word_name):
            word+=1
            namepool[word_name]=word
        else:
            word=namepool[word_name]
        label.append(word)
        namelist.append(path)
        for row in reader:
            if(row[0]!="untracked"):
                labelArr1.append(row)
                row2=[]
                for index in read_index:
                    row2.append(float(row[index*7]))
                    row2.append(float(row[index*7+1]))
                    row2.append(float(row[index*7+2]))
                data[data_index].append(row2)
            #else:
            #untrack[data_index]+=1
        assert(len(label)==len(data))
    return label,data,namelist
    
    


def construct_features(labels,data,namelist,bin_num=8,level_num=2,level_num_hog=3):
    assert len(labels)==len(data)
    #templates=load_templates("../data/hog_60template15mean.txt")
    #templates=load_templates("F:/study/save/generating/handshape/tmp.txt")
    ret_labels=[]
    features=[]
    velocity=[]
    height=[]
    for i in range(len(labels)):
        frames=data[i]
        if(i==127):
            i=127      
        if len(frames)==0:
            print i,labels[i],frames
            continue
        movement=hod(frames,bin_num,level_num)
        [movement_descriptor,right_v,right_h]=movement


        
#        deter_left=movement[1]
        #deter_right=movement[2]
        
        #hand_shape_descriptor=construct_hog_binary_features(frames,templates,1,level_num_hog,deter_left,deter_right)
#        if sum(hand_shape_descriptor)==0:
#            print i,'no length'
#            continue
        ret_labels.append(labels[i])
        features.append(normalize_histogram(movement_descriptor))
        velocity.append(right_v)
        height.append(right_h)
        #features.append(normalize_histogram(movement_descriptor)+normalize_histogram(hand_shape_descriptor))

    return ret_labels,features,velocity,height,namelist

def showMainImage(labelArr1,cap1,current_frame,tmp):
    cap1.set(1,current_frame)
    ret,frame1 = cap1.read()
    label = labelArr1[current_frame]
    prv_label = labelArr1[current_frame-1]
    #k = cv2.waitKey(0) & 0xFF
    if label[1]=="None":
        return 0
    if label[1]!="None":
        if label[1] == "Right" or label[1] == "Left" or label[1] == "Intersect":
            cnt = (float(label[2]),float(label[3])),(float(label[4]),float(label[5])),float(label[6])
            box = cv2.cv.BoxPoints(cnt)
            box = np.int0(box)
            output=subimage(frame1,(int(round(float(label[2]))),int(round(float(label[3])))),float(label[6])*3.14/180,int(round(float(label[4]))),int(round(float(label[5]))))
            outarr=asarray(output[:,:])

            for i in range(len(outarr)):
                for j in range(len(outarr[0])):
                    a=min(int(outarr[i,j,0]),int(outarr[i,j,1]),int(outarr[i,j,2]))[0][0]

                    outarr[i,j,:]=a


            ret,output2=threshold(outarr,float(label[7]),255,THRESH_BINARY_INV)

            #imwrite("/home/lzz/tmp/"+str(tmp)+".jpg",output2)
            #waitKey()
            #cv.SaveImage("/home/lzz/tmp/"+str(tmp)+".jpg",output)
            # waitKey()
            ''''t=[box[0][0],box[1][0],box[2][0],box[3][0]]
            left=box[0][0]
            for i in range(4):
                if(t[i]<left):
                    left=t[i]
            
            right=box[0][0]
            for i in range(4):
                if(t[i]>right):
                    right=t[i]
            
            t=[box[0][1],box[1][1],box[2][1],box[3][1]]
            up=box[0][1]
            for i in range(4):
                if(t[i]<up):
                    up=t[i]

            down=box[0][1]
            for i in range(4):
                if(t[i]>down):
                    down=t[i]
            subimage=frame1[up:down,left:right,:]
            subimageOri=frame[up:down,left:right,:]
            #subimage=frame1[300:414,300:400,:]
            M = getRotationMatrix2D((50,50),float(label[6]),1)
            dst = warpAffine(subimage,M,(200,200))
            dstOri = warpAffine(subimageOri,M,(200,200))

            height=round(sqrt((box[0][0]-box[1][0])**2+(box[0][1]-box[1][1])**2))
            width=round(sqrt((box[1][0]-box[2][0])**2+(box[1][1]-box[2][1])**2))
            

            t=dst[0:1,:][0]
            dstlist=dst.tolist()
            standard=[0,0,255]
            
            col=0
            while(1):
                try:
                    indend=dstlist[col].index(standard)
                    break
                except:
                    col+=1
                
                
            print indend
            imwrite("/home/lzz/tmp/"+str(tmp)+".jpg",dst[0:height-1,indend:indend+width])'''
           

        if label[1] == "Both":
            cnt1 = (float(label[2]),float(label[3])),(float(label[4]),float(label[5])),float(label[6])
            box1 = cv2.cv.BoxPoints(cnt1)
            box1 = np.int0(box1)
            output=subimage(frame1,(int(round(float(label[2]))),int(round(float(label[3])))),float(label[6])*3.14/180,int(round(float(label[4]))),int(round(float(label[5]))))
            outarr=asarray(output[:,:])

            for i in range(len(outarr)):
                for j in range(len(outarr[0])):
                    a=min(int(outarr[i,j,0]),int(outarr[i,j,1]),int(outarr[i,j,2]))[0][0]

                    outarr[i,j,:]=a


            ret,output2=threshold(outarr,float(label[12]),255,THRESH_BINARY_INV)            
            
            
            
            '''cv2.drawContours(frame1,[box1],0,(0,0,255),2)
            cv2.circle(frame1,(int(label[2]),int(label[3])),7,(0,0,255),-1)
            cv2.circle(frame1,(int(prv_label[2]),int(prv_label[3])),7,(100,100,250),-1)'''
            
            '''cnt2 = (float(label[8]),float(label[9])),(float(label[10]),float(label[11])),float(label[12])
            box2 = cv2.cv.BoxPoints(cnt2)
            box2 = np.int0(box2)
            cv2.drawContours(frame1,[box2],0,(0,0,255),2)
            cv2.circle(frame1,(int(label[8]),int(label[9])),7,(0,0,255),-1)
            cv2.circle(frame1,(int(prv_label[2]),int(prv_label[3])),7,(100,100,250),-1) '''
            
    #cv2.imshow('Video', merged_frame)
    #waitKey()
    return output2,label[0]

    
def list_file(path,filelist):
    files = os.listdir(path)
    for file in files:
        if(os.path.isdir(path+"/"+file)==1 and file[0:4]=="HKG_"):
            filelist.append(path+file)
        elif(os.path.isdir(path+"/"+file)==1 and file[0:4]!="HKG_"):
            list_file(path+"/"+file,filelist)
    return filelist
 
def subimage(image, centre, theta, width, height):
    output_image = cv.CreateImage((width, height), IPL_DEPTH_8U, 3)
    mapping = np.array([[np.cos(theta), -np.sin(theta), centre[0]],
                        [np.sin(theta), np.cos(theta), centre[1]]])
    map_matrix_cv = cv.fromarray(mapping)
    image2=cv.fromarray(image)
    cv.GetQuadrangleSubPix(image2, output_image, map_matrix_cv)
    return output_image


if __name__ == '__main__':
    #path='H:/Aaron/1-250/'
    path='/media/lzz/Data1/Aaron/1-250/'
    filelist=[]
    filelist=list_file(path,filelist)

    
    #filelist=["/media/lzz/Data1/Aaron/1-250/HKG_001_a_0026 Aaron 352"]
    [label,data,name]=load_data_no_mog(filelist)
    for i in range(len(label)):
        print label[i]
    [labels,features,velocity,height,name]=construct_features(label,data,name)
    
    '''v_file=open(filelist[0]+"/handshape/velocity.txt","w")
    for i in range(len(velocity)):
        
        v_file.write(str(i)+" "+str(velocity[i])+"\n")'''
        
        

    
    for l in range(len(labels)): 
        #csvfile = file(path+name[l]+'/feature.csv', 'wb')
        
        '''csvfile = file(name[l]+'/feature.csv', 'wb')
        writer = csv.writer(csvfile)
        data=[(features[l])]
        writer.writerows(data)
        csvfile.close()'''
        
        loc=name[l].rfind("/",0,len(name[l])-1)
        name2=name[l][loc:]
        Label1=open(name[l]+"/"+name2+'.csv','rb')
        vreader = csv.reader(Label1)
        untrack=0
        for rows in vreader:
            if(rows[0]=="untracked"):
                untrack+=1
            else:
                break
        
        
        
        
        if(os.path.exists(name[l]+"/handshape")==0):
            os.mkdir(name[l]+"/handshape")
        csvfile = file(name[l]+'/handshape/velocity.csv', 'wb')
        writer = csv.writer(csvfile)
        v=[]
        for i in range(1,len(velocity[l])):
            v.append([untrack+i+1,velocity[l][i],height[l][i]])
        writer.writerows(v) 
        csvfile.close()
            
            
            
    merged_frame = np.zeros((500,1190,3), np.uint8)
    for path in filelist:
        
        Label1=open(path+"/"+"label"+'.csv','rb')
        loc=path.rfind("/")
        name=path[loc:]
        cap1 = cv2.VideoCapture(path+name+"_d.avi")  
        reader = csv.reader(Label1)
        labelArr1 = []
        tmp=0
        for rows in reader:
            labelArr1.append(rows)
            ret=showMainImage(labelArr1,cap1,int(rows[0]),tmp)
            if(str(ret)=="0"):
                continue
            else:
                output,lab=ret
                imwrite(path+"/handshape/"+str(lab)+".jpg",output)
