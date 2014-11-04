'''
Created on Oct 29, 2014

@author: liuzz
'''
import csv
import cv2
import numpy
videoCapture = cv2.VideoCapture("D:/eclipse/project/save/newdata/documents-export-2014-10-28/c.avi")
fps = videoCapture.get(cv2.cv.CV_CAP_PROP_FPS)
size = (int(videoCapture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),int(videoCapture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
videoWriter = cv2.VideoWriter('carDemo.avi', cv2.cv.CV_FOURCC('I','4','2','0'), fps, size)
success, frame = videoCapture.read()

while success: # Loop until there are no more frames.
#    videoWriter.write(frame)
    f=numpy.array(frame)
    cv2.namedWindow("Image")  
    cv2.imshow("f",f)
    cv2.waitKey (0)  
    success, frame = videoCapture.read()

'''
csvfile = file("D:/eclipse/project/save/newdata/documents-export-2014-10-28/skeleton.csv", 'rb')
reader = csv.reader(csvfile)
data=[]
for line in reader:
    data.append(line)

csvfile.close()'''
