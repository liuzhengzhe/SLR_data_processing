#from Tkinter import *
import numpy as np
import cv2
import csv

#g_slider_position = 0

    
def nothing(x):
    pass

def showMainImage1(current_frame):
    #current_frame1 = cv2.getTrackbarPos("Silder1", "Video")
    cap1.set(1,current_frame)
    ret,frame1 = cap1.read()
    label = labelArr1[current_frame]
    prv_label = labelArr1[current_frame-1]
    #k = cv2.waitKey(0) & 0xFF
    if label[1]!="None":
        if label[1] == "Right" or label[1] == "Left" or label[1] == "Intersect":
            cnt = (float(label[2]),float(label[3])),(float(label[4]),float(label[5])),float(label[6])
            box = cv2.cv.BoxPoints(cnt)
            box = np.int0(box)
            cv2.drawContours(frame1,[box],0,(0,0,255),2)
            print label[2]
            cv2.circle(frame1,(int(float(label[2])),int(float(label[3]))),7,(0,0,255),-1)
            cv2.circle(frame1,(int(float(prv_label[2])),int(float(prv_label[3]))),7,(100,100,250),-1)
        
        if label[1] == "Both":
            cnt1 = (float(label[2]),float(label[3])),(float(label[4]),float(label[5])),float(label[6])
            box1 = cv2.cv.BoxPoints(cnt1)
            box1 = np.int0(box1)
            cv2.drawContours(frame1,[box1],0,(0,0,255),2)
            cv2.circle(frame1,(int(label[2]),int(label[3])),7,(0,0,255),-1)
            cv2.circle(frame1,(int(prv_label[2]),int(prv_label[3])),7,(100,100,250),-1)
            
            cnt2 = (float(label[8]),float(label[9])),(float(label[10]),float(label[11])),float(label[12])
            box2 = cv2.cv.BoxPoints(cnt2)
            box2 = np.int0(box2)
            cv2.drawContours(frame1,[box2],0,(0,0,255),2)
            cv2.circle(frame1,(int(label[8]),int(label[9])),7,(0,0,255),-1)
            cv2.circle(frame1,(int(prv_label[2]),int(prv_label[3])),7,(100,100,250),-1) 
             
    rsFrame1 = cv2.resize(frame1,(320,240))
    rsFrame1 = cv2.copyMakeBorder(rsFrame1,2,2,2,2,cv2.BORDER_CONSTANT,value=(0,0,255))
    merged_frame[:244, 10:334] = rsFrame1
    cv2.imshow('Video', merged_frame)
    return frame1
    
def showMainImage2(current_frame):
    #current_frame2 = cv2.getTrackbarPos("Silder1", "Video")
    cap2.set(1,current_frame)
    ret,frame2 = cap2.read()
#     skeleton = skeletonArr2[current_frame]
#     if skeleton[0]!="untracked":  
#         for i in range(0,10): #for i in range(0,14)
#             index_x = 5 + i*7
#             index_y = 6 + i*7
#             cv2.circle(frame2,(int(skeleton[index_x]),int(skeleton[index_y])),3,(0,0,255),-1)
    
    label = labelArr2[current_frame]
    prv_label = labelArr2[current_frame-1]
    if label[1]!="None":
        if label[1] == "Right" or label[1] == "Left" or label[1] == "Intersect":
            cnt = (float(label[2]),float(label[3])),(float(label[4]),float(label[5])),float(label[6])
            box = cv2.cv.BoxPoints(cnt)
            box = np.int0(box)
            cv2.drawContours(frame2,[box],0,(0,0,255),2)
            cv2.circle(frame2,(int(label[2]),int(label[3])),7,(0,0,255),-1)
            cv2.circle(frame2,(int(prv_label[2]),int(prv_label[3])),7,(100,100,250),-1)
        
        if label[1] == "Both":
            cnt1 = (float(label[2]),float(label[3])),(float(label[4]),float(label[5])),float(label[6])
            box1 = cv2.cv.BoxPoints(cnt1)
            box1 = np.int0(box1)
            cv2.drawContours(frame2,[box1],0,(0,0,255),2)
            cv2.circle(frame2,(int(label[2]),int(label[3])),7,(0,0,255),-1)
            cv2.circle(frame2,(int(prv_label[2]),int(prv_label[3])),7,(100,100,250),-1)
            
            cnt2 = (float(label[8]),float(label[9])),(float(label[10]),float(label[11])),float(label[12])
            box2 = cv2.cv.BoxPoints(cnt2)
            box2 = np.int0(box2)
            cv2.drawContours(frame2,[box2],0,(0,0,255),2)  
            cv2.circle(frame2,(int(label[2]),int(label[3])),7,(0,0,255),-1)
            cv2.circle(frame2,(int(prv_label[2]),int(prv_label[3])),7,(100,100,250),-1)
    #tempImage = cv2.cv.fromarray(frame2)
    #cv2.cv.DrawContours(tempImage, [box], (0,0,0), (255,255,255), 0)
    #im = np.asarray(tempImage)
    #h2,w2 = frame2.shape[:2]
    rsFrame2 = cv2.resize(frame2,(320,240))
    rsFrame2 = cv2.copyMakeBorder(rsFrame2,2,2,2,2,cv2.BORDER_CONSTANT,value=(0,0,255))
    merged_frame[254:498, 10:334] = rsFrame2
    cv2.imshow('Video', merged_frame)
    return frame2
#def rotateImage(image, angle):
    #image_center = tuple(np.array(image.shape)/2)
    #rot_mat = cv2.getRotationMatrix2D(image_center,angle,1.0)
    #result = cv2.warpAffine(image, rot_mat, image.shape,flags=cv2.INTER_LINEAR)
    #return result
def  showSkeleton(current_frame1,current_frame2,x):
        frame1 = showMainImage1(current_frame1)
        frame2 = showMainImage2(current_frame2)
        skeleton1 = skeletonArr1[current_frame1]
        skeleton2 = skeletonArr2[current_frame2]
        point =[]
        if skeleton1[0]!="untracked":
                for i in range(0,10): #for i in range(0,14)
                    index_x = 3 + i*7
                    index_y = 4 + i*7
                    point.append((int(skeleton1[index_x]),int(skeleton1[index_y])))
                    #draw line between points
                if x==1:
                    cv2.circle(frame1,point[0],3,(0,0,255),-1)
                elif x==2:
                    cv2.circle(frame1,point[1],3,(0,0,255),-1)
                    cv2.circle(frame1,point[2],3,(0,0,255),-1)
                    cv2.circle(frame1,point[3],3,(0,0,255),-1)
                    cv2.line(frame1,point[1],point[2],(0,0,255),1)
                    cv2.line(frame1,point[2],point[3],(0,0,255),1)
                elif x==3:
                    cv2.circle(frame1,point[4],3,(0,0,255),-1)
                    cv2.circle(frame1,point[5],3,(0,0,255),-1)
                    cv2.line(frame1,point[4],point[5],(0,0,255),1)
                elif x==4:
                    cv2.circle(frame1,point[6],3,(0,0,255),-1)
                    cv2.circle(frame1,point[7],3,(0,0,255),-1)
                    cv2.line(frame1,point[6],point[7],(0,0,255),1)
                elif x==5:
                    cv2.circle(frame1,point[8],3,(0,0,255),-1)
                    cv2.circle(frame1,point[9],3,(0,0,255),-1)
                    cv2.line(frame1,point[8],point[9],(0,0,255),1)
                    
        if skeleton2[0]!="untracked":
                for i in range(0,10): #for i in range(0,14)
                    index_x = 3 + i*7
                    index_y = 4 + i*7
                    point.append((int(skeleton2[index_x]),int(skeleton2[index_y])))
                    #draw line between points
                if x==1:
                    cv2.circle(frame2,point[0],3,(0,0,255),-1)
                elif x==2:
                    cv2.circle(frame2,point[1],3,(0,0,255),-1)
                    cv2.circle(frame2,point[2],3,(0,0,255),-1)
                    cv2.circle(frame2,point[3],3,(0,0,255),-1)
                    cv2.line(frame2,point[1],point[2],(0,0,255),1)
                    cv2.line(frame2,point[2],point[3],(0,0,255),1)
                elif x==3:
                    cv2.circle(frame2,point[4],3,(0,0,255),-1)
                    cv2.circle(frame2,point[5],3,(0,0,255),-1)
                    cv2.line(frame2,point[4],point[5],(0,0,255),1)
                elif x==4:
                    cv2.circle(frame2,point[6],3,(0,0,255),-1)
                    cv2.circle(frame2,point[7],3,(0,0,255),-1)
                    cv2.line(frame2,point[6],point[7],(0,0,255),1)
                elif x==5:
                    cv2.circle(frame2,point[8],3,(0,0,255),-1)
                    cv2.circle(frame2,point[9],3,(0,0,255),-1)
                    cv2.line(frame2,point[8],point[9],(0,0,255),1)
        
                
                #cv2.line(frame1,point[8],point[6],(0,0,255),1)

        
        rsFrame1 = cv2.resize(frame1,(320,240))
        rsFrame1 = cv2.copyMakeBorder(rsFrame1,2,2,2,2,cv2.BORDER_CONSTANT,value=(0,0,255))
        rsFrame2 = cv2.resize(frame2,(320,240))
        rsFrame2 = cv2.copyMakeBorder(rsFrame2,2,2,2,2,cv2.BORDER_CONSTANT,value=(0,0,255))
        merged_frame[:244, 10:334] = rsFrame1
        merged_frame[254:498, 10:334] = rsFrame2
        
        cv2.imshow('Video', merged_frame)  

def showBWImage(current_frame,i):
    #cnt = (389,263),(40,34),0
    #box = cv2.cv.BoxPoints(cnt)
    #box = np.int0(box)
    if i==1: 
        cap = cap1
        label = labelArr1[current_frame]
    else: 
        cap = cap2
        label = labelArr2[current_frame]
    #current_frame = cv2.getTrackbarPos("Silder1", "Video")
    cap.set(1,current_frame)
    ret,image = cap.read()
    if label[1]!="None":
        if label[1] == "Right" or label[1] == "Left" or label[1] == "Intersect":
            cnt = (float(label[2]),float(label[3])),(float(label[4]),float(label[5])),float(label[6])
            box = cv2.cv.BoxPoints(cnt)
            box = np.int0(box)
            mask = np.zeros((480,640,3), np.uint8)            
            cv2.fillPoly(mask, np.int32([box]), (255,255,255))            
            cv2.threshold(image,int(label[7]),255,cv2.THRESH_BINARY,image)            
            masked_image = cv2.bitwise_and(image, mask)
            cv2.drawContours(masked_image ,[box],0,(0,0,255),2)
            masked_image = cv2.resize(masked_image,(320,240))
            masked_image = cv2.copyMakeBorder(masked_image,2,2,2,2,cv2.BORDER_CONSTANT,value=(0,0,255))
            #merged_frame[:244, 344:668]= masked_image
    else:
        masked_image = np.zeros((240,320,3), np.uint8)
        masked_image = cv2.copyMakeBorder(masked_image,2,2,2,2,cv2.BORDER_CONSTANT,value=(0,0,255))
    if i==1: merged_frame[:244, 344:668]= masked_image 
    else: merged_frame[254:498, 344:668]= masked_image 
        #cv2.imshow("Hand Image: "+label[1], masked_image) 
            
#         #this is for the hand type "both"    
#         if label[1] == "Both":
#             cnt1 = (float(label[2]),float(label[3])),(float(label[4]),float(label[5])),float(label[6])
#             cnt2 = (float(label[8]),float(label[9])),(float(label[10]),float(label[11])),float(label[12])
#             box1 = cv2.cv.BoxPoints(cnt1)
#             box1 = np.int0(box1)
#             box2 = cv2.cv.BoxPoints(cnt2)
#             box2 = np.int0(box2)
#             mask1 = np.zeros((480,640,3), np.uint8)
#             mask2 = np.zeros((480,640,3), np.uint8)
#                 #mask = np.zeros((500,1310,3), np.uint8)
#             cv2.fillPoly(mask1, np.int32([box1]), (255,255,255))
#             cv2.fillPoly(mask2, np.int32([box2]), (255,255,255))
#             cv2.threshold(image,int(label[7]),255,cv2.THRESH_BINARY,image)
#              
#                 #mask[:480, 10:650] = mask1
#                 #mask[:480, 660:1300] = mask2
#             masked_image1 = cv2.bitwise_and(image, mask1)
#              
#             masked_image2 = cv2.bitwise_and(image, mask2)
#             cv2.drawContours(masked_image1,[box1],0,(0,0,255),2)
#             cv2.drawContours(masked_image2,[box2],0,(0,0,255),2)
#             masked_image1 = cv2.resize(masked_image1,(320,240))
#             #merged_frame[:240, 10:330]= frame2
#             cv2.imshow("Hand Image: Right", masked_image1) 
#             cv2.imshow("Hand Image: Left", masked_image2) 


def showHogImage(current_frame,i):
    if i==1: dir = ''.join(['D:/CUHK_Project/HandDetector/sample1/visualization/',str(current_frame),'.jpg'])
    else: dir = ''.join(['D:/CUHK_Project/HandDetector/sample2/visualization/',str(current_frame),'.jpg'])
    hogImage = cv2.imread(dir)
    if hogImage==None : hogImage = np.zeros((240,240,3), np.uint8)
    hogImage = cv2.resize(hogImage,(240,240))
    hogImage = cv2.copyMakeBorder(hogImage,2,2,2,2,cv2.BORDER_CONSTANT,value=(0,0,255))
    if i==1: merged_frame[:244, 678:922]= hogImage
    else: merged_frame[254:498, 678:922]= hogImage

'''def showKeyImage(i):
    if i== 1: dir = 'D:/CUHK_Project/HandDetector/sample1/visualization/40.jpg'
    else: dir = 'D:/CUHK_Project/HandDetector/sample2/visualization/40.jpg'
    keyImage = cv2.imread(dir)
    keyImage = cv2.resize(keyImage,(240,240))
    keyImage = cv2.copyMakeBorder(keyImage,2,2,2,2,cv2.BORDER_CONSTANT,value=(0,0,255))
    if i==1: merged_frame[:244, 932:1176]= keyImage
    else: merged_frame[254:498, 932:1176]= keyImage'''

def silder1_display(current_frame):
    current_frame = cv2.getTrackbarPos("Silder1", "Video")
    showBWImage(current_frame,1)
    showHogImage(current_frame,1)
    #showKeyImage(1)
    showMainImage1(current_frame) 

def silder2_display(current_frame):
    current_frame = cv2.getTrackbarPos("Silder2", "Video")
    showBWImage(current_frame,2)
    showHogImage(current_frame,2)
    #showKeyImage(2)
    showMainImage2(current_frame)
    
     
#merged_frame = cv.CreateImage((1500,500), 1, 1)
#cnt = (389,263),(40,34),0
#box = cv2.cv.BoxPoints(cnt)
#box = np.int0(box)
#cv2.namedWindow("image")
if __name__ == '__main__':
    cap1 = cv2.VideoCapture("/media/lzz/Data1/Aaron/1-250/HKG_001_a_0001 Aaron 22/HKG_001_a_0001 Aaron 22_d.avi")  
    cap2 = cv2.VideoCapture("/media/lzz/Data1/Aaron/1-250/HKG_001_a_0001 Aaron 22/HKG_001_a_0001 Aaron 22_d.avi")
    merged_frame = np.zeros((500,1190,3), np.uint8)
    cv2.namedWindow("Video",cv2.CV_WINDOW_AUTOSIZE)
    with open('/media/lzz/Data1/Aaron/1-250/HKG_001_a_0001 Aaron 22/label.csv','rb') as Label1:
        reader = csv.reader(Label1)
        labelArr1 = []
        for row in reader : labelArr1.append(row)
    with open('/media/lzz/Data1/Aaron/1-250/HKG_001_a_0001 Aaron 22/label.csv','rb') as Label2:
        reader = csv.reader(Label2)
        labelArr2 = []
        for row in reader : labelArr2.append(row)
    with open('/media/lzz/Data1/Aaron/1-250/HKG_001_a_0001 Aaron 22/HKG_001_a_0001 Aaron 22.csv','rb') as Skeleton1:
        reader = csv.reader(Skeleton1)
        skeletonArr1 = []
        for row in reader : skeletonArr1.append(row)
    with open('/media/lzz/Data1/Aaron/1-250/HKG_001_a_0001 Aaron 22/HKG_001_a_0001 Aaron 22.csv','rb') as Skeleton2:
        reader = csv.reader(Skeleton2)
        skeletonArr2 = []
        for row in reader : skeletonArr2.append(row)
#no_of_frames = int(cv2.GetCaptureProperty(cap,7))
    no_of_frames1 = int(cap1.get(7))
    no_of_frames2 = int(cap2.get(7))
    #switch = '0 : CLOSE \n1 : OPEN'
    if no_of_frames1 != 0 and no_of_frames2!=0:
        cv2.createTrackbar("Silder1", "Video",0, no_of_frames1, silder1_display)
        cv2.createTrackbar("Silder2", "Video",0, no_of_frames2, silder2_display)
        #switch = 'show skeleton'
        #cv2.createTrackbar('switch',"Video",0,1,nothing)
        #cv2.createTrackbar(switch, 'Video',0,1,openImage)
        
    while cap1.isOpened() and cap2.isOpened() :
    #while 1:
        k = cv2.waitKey(0) & 0xFF
        current_frame1 = cv2.getTrackbarPos("Silder1", "Video")
        current_frame2 = cv2.getTrackbarPos("Silder2", "Video")
        if k == 49:
            showSkeleton(current_frame1, current_frame2,1)
        if k == 50:
            showSkeleton(current_frame1, current_frame2,2) 
        if k == 51:
            showSkeleton(current_frame1, current_frame2,3)
        if k == 52:
            showSkeleton(current_frame1, current_frame2,4)
        if k == 53:
            showSkeleton(current_frame1, current_frame2,5)
        if k == 27:
            break
    cap1.release()    
    cap2.release()
    cv2.destroyAllWindows()
        
    
        