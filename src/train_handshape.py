'''
Created on Nov 14, 2014

@author: liuzz
'''
from cv2 import *
import cv2
import os
from basic import *
import time
from constant_numbers import *
import struct
from load import *
from kMedoid import *
from numpy import *
    
if __name__ == '__main__':
    path="D:/caffe/caffe-windows/caffe-windows/handimages/handimages/"
    path2="D:/caffe/caffe-windows/caffe-windows/examples/mnist/handshapes/"
    f1=open(path2+"trainold.txt","w")
    f2=open(path2+"testold.txt","w")
    #list0=[[37,1550],[1551,2924],[2925,4317],[4318,5052],[5053,5589],[5590,6289],[6312,6869],[6873,9055]]
    for ran in range (0,13):
        os.mkdir(path2+"train/"+str(ran))
        os.mkdir(path2+"test/"+str(ran))
    for ran in range (0,13):
        files = os.listdir(path+str(ran))
        #for i in range(int(list0[ran][0]),int(list0[ran][1])):
        cnt=0
        for f in files:
            #img=imread(path+str(i)+"_r.jpg")
            cnt+=1
            img=imread(path+str(ran)+"/"+f)
            sp=img.shape
            img2=copyMakeBorder(img, 0,0, int(abs(sp[0]-sp[1])/2),int(abs(sp[0]-sp[1])/2), BORDER_CONSTANT, value=(0, 0, 0, 0))
            img3=cv2.resize(img2,(128,128))
            if(random.random()<0.9):
                
                imwrite(path2+"train/"+str(ran)+"/"+f,img3)
                f1.write("train/"+str(ran)+"/"+str(f)+" "+str(ran)+"\n")
            else:
                
                imwrite(path2+"test/"+str(ran)+"/"+f,img3)
                f2.write("test/"+str(ran)+"/"+str(f)+" "+str(ran)+"\n")
            
            
            
            '''row=cv.GetSize(img)
            col=cv.GetSize(img,1)
            im2=hconcat(img, img)
            imshow("1",im2)
            waitKey()'''
