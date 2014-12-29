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
import matplotlib.pyplot as plt
import cv2
from skimage.feature import hog
from skimage import data, color, exposure
import skimage
#path="/media/lzz/Data1/Aaron/1-250/HKG_001_a_0001 Aaron 11"
path="/home/lzz/caffe-master/new/handshapes/train/13/2.jpg"
image=cv2.imread(path)
image = color.rgb2gray(image)

fd, hog_image = hog(image, orientations=9, pixels_per_cell=(16,16),cells_per_block=(2, 2), visualise=True)
print len(fd)
for i in range(len(fd)):
    print fd[i]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

ax1.axis('off')
ax1.imshow(image)
ax1.set_title('Input image')

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.002))

ax2.axis('off')
ax2.imshow(hog_image_rescaled)
ax2.set_title('Histogram of Oriented Gradients')
plt.show()