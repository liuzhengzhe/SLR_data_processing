'''import cv2
im = cv2.imread("/home/lzz/1.jpg",cv2.CV_LOAD_IMAGE_COLOR)
#im = cv2.cv.LoadImage("/home/lzz/1.jpg",cv2.CV_LOAD_IMAGE_COLOR)
cv2.imshow("1",im)
cv2.waitKey()'''

'''import caller

if __name__=="__main__":
    def callback(respose_data):
        print respose_data
    caller.ds_asyncore(('192.168.212.96', 8080),callback,timeout=5)'''
'''def my_callback(input):
    print "function my_callback was called with %s input" % (input,)
 
def caller(input, func):
    func(input)
 
for i in range(5):
    caller(i, my_callback)'''
'''
l=["a","s","a","s","a","s","x"]
a=bytearray(l)    
print len(a)
b=a.split("s")
c=b[len(b)-1]
print c'''
'''from svmutil import *
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

from constant_numbers import *
import os
path="/media/lzz/Data1/Aaron/1-250/HKG_005_d_0009 Aaron 2081/handshape"
os.chdir(path)
print os.path.isfile(path+"/handshape/73.jpg")
os.rename("73.jpg","73*.jpg")'''
print float(5)/float(2)