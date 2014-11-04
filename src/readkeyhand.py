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
'''
s = os.sep
root = "D:/eclipse/project/save/caffe/"

def func(args,dire,fis):
    for f in fis:
        fname = os.path.splitext(f) 
        #new = fname[0] + 'b' + fname[1]
        #os.rename(os.path.join(dire,f),os.path.join(dire,new))
        img=imread("D:/eclipse/project/save/caffe/"+fname+"/"+"*r.jpg")
        img=imread("D:/eclipse/project/save/caffe/"+fname+"/"+str(best_choice+25)+"_r.jpg")
        imwrite("D:/eclipse/project/save/generating/caffe/generate/"+name.encode("utf-8")+".jpg",img)

os.path.walk(root,func,())'''

import os
allFileNum = 0
def printPath(level, path,f1,f2):
    global allFileNum

    dirList = []

    fileList = []

    files = os.listdir(path)

#    dirList.append(str(level))
    for f in files:
        if(os.path.isdir(path + '/' + f)):

            if(f[0] == '.'):
                pass
            else:

                dirList.append(f)

    i_dl = 0
    index=0
    for dl in dirList:
        '''if(i_dl == 0):
            i_dl = i_dl + 1
        else:

            print '-' * (int(dirList[0])), dl

            printPath((int(dirList[0]) + 1), path + '/' + dl)'''
        blogword=zeros((1,3600))
        dstimg= []
        path2=path+"/"+dl+"/"
        files = os.listdir(path2)
        flag=0
        for f in files:
#            if(os.path.isfile(path + '/' + f)):
                flag=1
                fileList.append(f)
                img=imread(path2+f)
                l0 = cv2.split(img)[0]
                l1 = cv2.split(img)[1]
                l2 = cv2.split(img)[2]
                b=cv2.resize(l0,(60,60))
                ''''g=resize(l1,[60,60])
                r=resize(l2,[60,60])
                img2 = cv2.merge([b,g,r])'''
                #imshow("1",b)
                #waitKey()
                img2=b
                img3=resize(img2,[1,3600])
                blogword=concatenate((blogword,img3))
                dstimg.append(img2)
        if(flag==0):
            #index+=1
            continue
        blogword=blogword[1:size(blogword),:]
        best_choice = kmedoids(blogword,1)
        index+=1
        imwrite('D:/eclipse/project/save/generating/caffe/key/'+str(index)+".jpg",dstimg[best_choice])
        f1.write(str(index)+" "+str(dl)+"\n")
#        imwrite("D:/eclipse/project/save/generating/caffe/generate/"+name.encode("utf-8")+".jpg",img)


if __name__ == '__main__':
    f1=open("D:/eclipse/project/save/generating/caffe/dic.txt","w")
    f2=open("D:/eclipse/project/save/generating/caffe/save.txt")
    printPath(1, 'D:/eclipse/project/save/caffe',f1,f2)
    print 'number =', allFileNum
