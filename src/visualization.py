'''
Created on Oct 15, 2014

@author: liuzz
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
from hogmodule import *
from svmmodule import *
from cv2 import *
from math import *



def get_hogdescriptor_visual_image(velist,indexsave,name,signer):

    cell_mat=[[[] for col in range(11)] for row in range(11)]
    for j in range(121):
                r=[];
                for q in range(9):
                    r.append(velist[j*36+q])
#                print "r",r;
                
                cell_mat[j%11][int(j/11)].append(r)
                r=[]
                for q in range(9,18):
                    r.append(velist[j*36+q])
#                print "r",r;
                cell_mat[j%11][int(j/11)].append(r)
                r=[]
                for q in range(18,27):
                    r.append(velist[j*36+q])
#                print "r",r;
                cell_mat[j%11][int(j/11)].append(r)
                r=[]
                for q in range(27,36):
                    if(j*36+q==4356):
                        j
                    r.append(velist[j*36+q])
                cell_mat[j%11][int(j/11)].append(r)

    block_mat=[[[0 for dep in range(9)] for col in range(12)] for row in range(12)]
    weight=[[0 for col in range(12)] for row in range(12)]
        
    for i in range(11):
                for j in range(11):
                    for x in range(9):
                        block_mat[i][j][x]+=cell_mat[i][j][0][x];
                        block_mat[i+1][j][x]+=cell_mat[i][j][1][x];
                        block_mat[i][j+1][x]+=cell_mat[i][j][2][x];
                        block_mat[i+1][j+1][x]+=cell_mat[i][j][3][x];
                    weight[i][j]=weight[i][j]+1;
                    weight[i][j+1]+=1;
                    weight[i+1][j]+=1;
                    weight[i+1][j+1]+=1;
    for i in range(12):
                for j in range(12):
                    for x in range(9):
                        block_mat[i][j][x]/=weight[i][j]
                a=str(0)
    for i in range(1,90000*3):
        a=a+"0"
    
    randomByteArray = bytearray(a)

    flatNumpyArray = numpy.array(randomByteArray)
    img = zeros([300,300,3])
#    img = flatNumpyArray.reshape(300, 300,3)
    for y in range(12):
        for x in range(12):
            drawx=25*x
            drawy=25*y
            mx=drawx+12.5
            my=drawy+12.5
            a=rectangle(img,(drawy,drawx),(drawy+25,drawx+25),(0,255,0))
        
        
            for b in range(9):
                if (block_mat[y][x][b]==0):
                    continue
                rad=b*3.14/9+3.14/18
                
                dirx=cos(rad)
                diry=sin(rad)
                x1=mx-dirx*block_mat[y][x][b]*12.5*500
                y1=my-diry*block_mat[y][x][b]*12.5*500
                x2=mx+dirx*block_mat[y][x][b]*12.5*500
                y2=my+diry*block_mat[y][x][b]*12.5*500
                line(img,
                     (int(x1),int(y1)),
                     (int(x2),int(y2)),
                     (255,255,255));
                
        
        
        
        
        
        
        
        
        
        
        
    imwrite("D:/eclipse/project/save/visualization/"+name+"_"+signer+"_"+str(indexsave)+".jpg",img)    
      
    


if __name__ == '__main__':

    ifs=open("D:/eclipse/project/save/generating/handshape/tmp.txt");
    velist=[]
    db=sqlite3.connect("../data/features.db")
    cu=db.cursor()
    cu.execute("Select data,name,signer from Aaron_1_50_1_1")
    data0=cu.fetchall()
    data=[]
    for i in range(len(data0)):
        ptemp=data0[i][0].replace("\\n","\n")
        ptemp=str(ptemp);
        t2 = pickle.loads(ptemp);
        t3=t2[192:4549]
        namei=data0[i][1].encode("utf-8")
        signeri=data0[i][2].encode("utf-8")
        get_hogdescriptor_visual_image(t3,i,namei,signeri)
        
        
#        data.append(t2);    
        
        
    

















