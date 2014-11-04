'''
Created on 2014-9-3

@author: lenovo
'''
from svmutil import *
from svmmodule import *
import sqlite3
import math
import numpy
import struct
from hmm.continuous.GMHMM import GMHMM
from cluster import KMeansClustering
import time
import random
#import matplotlib.pyplot as plt1  
import marshal, pickle
import cv2
from numpy import *
import numpy as np
LSX=0
LSY=1
LSZ=2
RSX=3
RSY=4
RSZ=5
LEX=6
LEY=7
LEZ=8
REX=9
REY=10
REZ=11
LHX=12
LHY=13
LHZ=14
RHX=15
RHY=16
RHZ=17
HX=18
HY=19
HZ=20
HIPCX=21
HIPCY=22
HIPCZ=23
LWX=24
LWY=25
LWZ=26
RWX=27
RWY=28
RWZ=29
LEFT_HOG=30
RIGHT_HOG=31
LEFT_MOG=32
RIGHT_MOG=33
RAW_DATA_LENGTH=34
MIN_FRAME_NUM=20
INIT_THRESHOLD=0.3
MOG_LENGTH=24
FEATURE_SIZE_PER_FRAME=20#+2*MOG_LENGTH
TEMPLATE_THRESHOLD=0.165
label2classNo={}

def hog_distance(h1,h2):
    assert len(h1)==len(h2)
    ret=0
    for i in range(0,len(h1)):
        ret+=((h1[i]-h2[i])**2)
#     print ret
    ret=math.sqrt(ret/len(h1))
#     if ret<TEMPLATE_THRESHOLD:
#         print ret  
    return ret

def load_templates(template_file="../data/hog_template5.txt"):
    ret=[]
    templates=open(template_file)
    for t in templates:
        ret.append([float(i) for i in t.split()])
#     random.shuffle(ret)
#     print len(ret)
#     ret=ret[0:5]+ret[30:35]+ret[285:290]
    return ret

def load_handshapes(db_file_name="../data/Aaron1-50.db"):
    ret=[]
    db = sqlite3.connect(db_file_name)
    for hog in db.execute("select LeftHandHOG from framedata;"):
        if hog[0]:
            new_hog= tuple([struct.unpack('f',hog[0][i:i+4])[0] for i in range(0,len(hog[0]),4)])
            ret.append(new_hog)
            #print len(new_hog)
    for hog in db.execute("select rightHandHOG from framedata;"):
        if hog[0]:
            new_hog=tuple( [struct.unpack('f',hog[0][i:i+4])[0] for i in range(0,len(hog[0]),4)])
            ret.append(new_hog)
    return ret

def hog(hand_data):
    new_hog=tuple( [struct.unpack('f',hand_data[i:i+4])[0] for i in range(0,len(hand_data),4)])
#     print new_hog
    return new_hog


def load_data_no_mog(db_file_name="../data/data7.db"):
    RAW_DATA_LENGTH=32
    db = sqlite3.connect(db_file_name)
    labels=[]
    #label2classNo={}
    classNo2Label={}
    classNo=0
    current_data_index=-1
    current_sign_id=0
    current_class_id=0
    data=[]
    for sign_id,frame_id,word_id,lsx,lsy,lsz,rsx,rsy,rsz,lex,ley,lez,rex,rey,lez,lhx,lhy,lhz,rhx,rhy,rhz,hx,hy,hz,hipcx,hipcy,hipcz,lwx,lwy,lwz,rwx,rwy,rwz,leftHog,rightHog in db.execute("Select FrameData.SampleIndex,FrameData.index_ID,signId,SkeletonShoulderLeftX,SkeletonShoulderLeftY,SkeletonShoulderLeftZ,SkeletonShoulderRightX,SkeletonShoulderRightY,SkeletonShoulderRightZ,SkeletonElbowLeftX,SkeletonElbowLeftY,SkeletonElbowLeftZ,SkeletonElbowRightX,SkeletonElbowRightY,SkeletonElbowRightZ,SkeletonHandLeftX,SkeletonHandLeftY,SkeletonHandLeftZ,SkeletonHandRightX,SkeletonHandRightY,SkeletonHandRightZ,SkeletonHeadX,SkeletonHeadY,SkeletonHeadZ,SkeletonHipCenterX,SkeletonHipCenterY,SkeletonHipCenterZ,SkeletonWristLeftX,SkeletonWristLeftY,SkeletonWristLeftZ,SkeletonWristRightX,SkeletonWristRightY,SkeletonWristRightZ,LeftHandHOG,RightHandHog from FrameData, SignSample where FrameData.SampleIndex=SignSample.index_ID ORDER BY  FrameData.index_ID,FrameNumber;"):     
        if current_class_id!=word_id and (not label2classNo.has_key(word_id)):
            classNo+=1
            current_class_id=word_id
            label2classNo[word_id]=classNo
            classNo2Label[classNo]=word_id
        
        if current_sign_id!=sign_id:
            #new sign record
            if current_data_index>=0 and len(data[current_data_index])<54:
                print '!!!!',current_sign_id,len(data[current_data_index])
            current_sign_id=sign_id
            data.append(list())
            labels.append(label2classNo[word_id]) 
            current_data_index+=1
#         print current_sign_id,len(data)
        if lsx:
            data[current_data_index].append([lsx,lsy,lsz,rsx,rsy,rsz,lex,ley,lez,rex,rey,lez,lhx,lhy,lhz,rhx,rhy,rhz,hx,hy,hz,hipcx,hipcy,hipcz,lwx,lwy,lwz,rwx,rwy,rwz,leftHog,rightHog])
#         print data[current_data_index]
    return labels,data,classNo2Label

def load_data(db_file_name="../data/data7.db"):
    db = sqlite3.connect(db_file_name)
    labels=[]
    label2classNo={}
    classNo2Label={}
    classNo=0
    current_data_index=-1
    current_sign_id=0
    current_class_id=0
    data=[]
    for sign_id,frame_id,word_id,lsx,lsy,lsz,rsx,rsy,rsz,lex,ley,lez,rex,rey,lez,lhx,lhy,lhz,rhx,rhy,rhz,hx,hy,hz,hipcx,hipcy,hipcz,lwx,lwy,lwz,rwx,rwy,rwz,leftHog,rightHog,leftMog,rightMog in db.execute("Select FrameData.SampleIndex,FrameData.index_ID,signId,SkeletonShoulderLeftX,SkeletonShoulderLeftY,SkeletonShoulderLeftZ,SkeletonShoulderRightX,SkeletonShoulderRightY,SkeletonShoulderRightZ,SkeletonElbowLeftX,SkeletonElbowLeftY,SkeletonElbowLeftZ,SkeletonElbowRightX,SkeletonElbowRightY,SkeletonElbowRightZ,SkeletonHandLeftX,SkeletonHandLeftY,SkeletonHandLeftZ,SkeletonHandRightX,SkeletonHandRightY,SkeletonHandRightZ,SkeletonHeadX,SkeletonHeadY,SkeletonHeadZ,SkeletonHipCenterX,SkeletonHipCenterY,SkeletonHipCenterZ,SkeletonWristLeftX,SkeletonWristLeftY,SkeletonWristLeftZ,SkeletonWristRightX,SkeletonWristRightY,SkeletonWristRightZ,LeftHandHOG,RightHandHog,MoGLeft,MoGRight from FrameData, SignSample where FrameData.SampleIndex=SignSample.index_ID ORDER BY  FrameData.index_ID,FrameNumber;"):     
        if current_class_id!=word_id and (not label2classNo.has_key(word_id)):
            classNo+=1
            current_class_id=word_id
            label2classNo[word_id]=classNo
            classNo2Label[classNo]=word_id
        
        if current_sign_id!=sign_id:
            #new sign record
            if current_data_index>=0 and len(data[current_data_index])<54:
                print '!!!!',current_sign_id,len(data[current_data_index])
            current_sign_id=sign_id
            data.append(list())
            labels.append(label2classNo[word_id]) 
            current_data_index+=1
#         print current_sign_id,len(data)
        if lsx:
            data[current_data_index].append([lsx,lsy,lsz,rsx,rsy,rsz,lex,ley,lez,rex,rey,lez,lhx,lhy,lhz,rhx,rhy,rhz,hx,hy,hz,hipcx,hipcy,hipcz,lwx,lwy,lwz,rwx,rwy,rwz,leftHog,rightHog,leftMog,rightMog])
#         print data[current_data_index]
    return labels,data,classNo2Label
#     import copy
#     o_frames=copy.deepcopy(data[0])
#     frames= smoothen_ali_and_ar(data)[0]
#     for i in range(0,len(frames)):
#         frame=frames[i]
#         o_frame=o_frames[i]
#         print frame[ALI1],o_frame[ALI1],frame[AR1],o_frame[AR1],frame[ALI2],o_frame[ALI2],frame[AR2],o_frame[AR2]
    
#     for i in range(0,len(data)):
#         if labels[i]!=-1:
#             print classNo2Label[labels[i]],labels[i],data[i]
#     print len(labels),len(data)    
#     print sign_id,frame_id,word_id,lsx,lsy,rsx,rsy,lex,ley,rex,rey,lhx,lhy,rhx,rhy
def get_possible_region(features):
    distance_head_to_hip=features[0][HIPCY]-features[0][HY]
    print distance_head_to_hip
    x1=features[0][HX]-distance_head_to_hip*0.5
    z1=features[0][HY]-distance_head_to_hip*1.2
    x2=features[0][HIPCX]+distance_head_to_hip*0.5
    z2=features[0][HIPCY]  
    return x1,z1,x2,z2
    
def hop_2d(features,edge_bin_num=3):
    x1,z1,x2,z2=get_possible_region(features)
    left_hand_bins=[[0]*edge_bin_num]*edge_bin_num
    right_hand_bins=[[0]*edge_bin_num]*edge_bin_num
    left_no_repeat=set()
    right_no_repeat=set()
    for feature in features:
        x=feature[LHX]
        z=feature[LHY]
        if (not ( (x,z) in left_no_repeat)) and x>x1 and x<x2 and z>z1 and z<z2:
            bin_index_x,bin_index_y=get_hop_bin_for_a_point(x1,z1,x2,z2,x,z,edge_bin_num)
            left_hand_bins[bin_index_x][bin_index_y]+=1
        left_no_repeat.add((x,z))
        x=feature[RHX]
        z=feature[RHY]
        if (not (x,z) in right_no_repeat) and x>x1 and x<x2 and z>z1 and z<z2:
            bin_index_x,bin_index_y=get_hop_bin_for_a_point(x1,z1,x2,z2,x,z,edge_bin_num)
            right_hand_bins[bin_index_x][bin_index_y]+=1
        right_no_repeat.add((x,z))
    left_hand_bins=normalize_histogram(convert_2d_to_1d(left_hand_bins))
    right_hand_bins=normalize_histogram(convert_2d_to_1d(right_hand_bins))
    print left_hand_bins
    print right_hand_bins
    return left_hand_bins+right_hand_bins

def convert_2d_to_1d(bins):
    a=numpy.array(bins)
    ret= list(a.flatten())
    return ret

def get_hop_bin_for_a_point(x1,y1,x2,y2,x,y,edge_bin_num):
    print x1,y1,x2,y2,x,y
    x_i=int((x-x1)/(x2-x1)*edge_bin_num)
    y_i=int((y-y1)/(y2-y1)*edge_bin_num)
    print "index",x_i,y_i
    return x_i,y_i
    
    
def hod(features,bin_num=8,level_num=4):
    his=[]
    his+=hod_of_a_joint(features,LHX,LHY,LHZ,bin_num,level_num)
    his+=hod_of_a_joint(features,LSX,LSY,LSZ,bin_num,level_num)
    his+=hod_of_a_joint(features,LEX,LEY,LEZ,bin_num,level_num)
    his+=hod_of_a_joint(features,RHX,RHY,RHZ,bin_num,level_num)
    his+=hod_of_a_joint(features,RSX,RSY,RSZ,bin_num,level_num)
    his+=hod_of_a_joint(features,REX,REY,REZ,bin_num,level_num)
    his+=hod_of_a_joint(features,LWX,LWY,LWZ,bin_num,level_num)
    his+=hod_of_a_joint(features,RWX,RWY,RWZ,bin_num,level_num)

#     print len(his),his
    return his

def hod_of_a_joint(features,x_index,y_index,z_index,bin_num,level_num):  
    his=get_hod_for_a_node_of_pyramid(features,x_index,y_index,z_index,bin_num,1,level_num)              
    return his

def normalize_histogram(bins):
    total=sum(bins)
    if total==0:
        return [0.0]*len(bins)
    for i in range(0,len(bins)):
        bins[i]=float(bins[i])/float(total)
#     print bins
    return bins
    

def get_hod_for_a_node_of_pyramid(features,x_index,y_index,z_index,bin_num,level_index,level_num):
    if level_index==level_num:
        return []
    px=features[0][x_index]
    py=features[0][y_index]
    pz=features[0][z_index]
    bin_xy=[0]*bin_num
    bin_xz=[0]*bin_num
    bin_yz=[0]*bin_num
    for i in range(1,len(features)):
        x=features[i][x_index]
        y=features[i][y_index]
        z=features[i][z_index]
        v_xy=(y-py,x-px)
        v_xz=(z-pz,x-px)
        v_yz=(z-pz,y-py)
        bin_xy[get_bin_for_vec(v_xy,bin_num)]+=length(v_xy)
        bin_xz[get_bin_for_vec(v_xz,bin_num)]+=length(v_xz)
        bin_yz[get_bin_for_vec(v_yz,bin_num)]+=length(v_yz)
        
#         print bin_xz
        px=x
        py=y
        pz=z
    left_hod=get_hod_for_a_node_of_pyramid(features[0:len(features)/2],x_index,y_index,z_index,bin_num,level_index+1,level_num)
    right_hod=get_hod_for_a_node_of_pyramid(features[len(features)/2+1:len(features)],x_index,y_index,z_index,bin_num,level_index+1,level_num)
    bin_xy=normalize_histogram(bin_xy)
    bin_xz=normalize_histogram(bin_xz)
    bin_yz=normalize_histogram(bin_yz)
    return bin_xy+bin_xz+bin_yz+left_hod+right_hod

def get_bin_for_vec(vec_2d,bin_num):
    if vec_2d==(0,0):
        return 0
    ang=angle(vec_2d,(1,0))
    if vec_2d[1]<0:
        ang=2*math.pi-ang
    ret= int(ang/(2*math.pi)*bin_num)
    return ret

# def smoothen_hand_count_and_tip_count(data):
#     for frames in data:
#         for frame in frames:
#             if frame[HAND_COUNT] is None:
#                 frame[HAND_COUNT]=0
#             if frame[TIP_COUNT1] is None:
#                 frame[TIP_COUNT1]=-1
#             if frame[TIP_COUNT2] is None:
#                 frame[TIP_COUNT2]=-1
#     return data

# def smoothen_ali_and_ar(data):
#     data=smoothen_ali_and_ar_for_a_column(data,ALI1)
#     data=smoothen_ali_and_ar_for_a_column(data,AR1)     
#     data=smoothen_ali_and_ar_for_a_column(data,ALI2)     
#     data=smoothen_ali_and_ar_for_a_column(data,AR2)     
#     return data

def smoothen_mogs(data):
    data=smoothen_ali_and_ar_for_a_column(data,LEFT_MOG)
    data=smoothen_ali_and_ar_for_a_column(data,RIGHT_MOG)      
    return data

def smoothen_ali_and_ar_for_a_column(data,raw_data_column_index):
    for frames in data:
        mark=0
        for f in frames:
            if (frames[0][raw_data_column_index] is None) and (f[raw_data_column_index] is not None):
                frames[0][raw_data_column_index]=f[raw_data_column_index]
        
        for i in range(0,len(frames)):
            row=frames[i]
            if row[raw_data_column_index] and i-mark>0:
#                 print row[raw_data_column_index],x
                if frames[mark][raw_data_column_index] is not None:
                    delta=(row[raw_data_column_index]-frames[mark][raw_data_column_index])/float(i-mark)
                for j in range(mark+1,i):
                    if frames[mark][raw_data_column_index] is None:
                        frames[j][raw_data_column_index]=row[raw_data_column_index]
                    else:
                        frames[j][raw_data_column_index]=frames[mark][raw_data_column_index]+delta*(j-mark)
                mark=i
            if i==len(frames)-1 and frames[i][raw_data_column_index] is None:
                for j in range(mark+1,len(frames)):
                    frames[j][raw_data_column_index]=frames[mark][raw_data_column_index]               
    return data

def dotproduct(v1, v2):
    return sum((a*b) for a, b in zip(v1, v2))

def length(v):
    return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
#     print v1, v2
    return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))

def distance(x1,y1,z1,x2,y2,z2):
    return math.sqrt((x1-x2)**2+(y1-y2)**2+(z1-z2)**2)

def distance_2d(x1,y1,x2,y2):
    return math.sqrt((x1-x2)**2+(y1-y2)**2) 
 
def get_unit_vector(x1,y1,z1,x2,y2,z2):
#     print x1,y1,x2,y2
    if x1==x2 and y1==y2 and z1==z2:
        return 0,0
    return (float(x1-x2))/distance(x1,y1,z1,x2,y2,z2),(float(y1-y2))/distance(x1,y1,z1,x2,y2,z2),(float(z1-z2))/distance(x1,y1,z1,x2,y2,z2)

def get_2d_unit_vector(x1,y1,x2,y2):
#     print x1,y1,x2,y2
    if x1==x2 and y1==y2:
        return 0,0
    dis=distance_2d(x1,y1,x2,y2)
    return (float(x1-x2))/dis,(float(y1-y2))/dis


def get_vector(x1,y1,z1,x2,y2,z2,scale):
#     print x1,y1,x2,y2
    if x1==x2 and y1==y2 and z1==z2:
        return 0,0
    return (float(x1-x2)/float(scale)),(float(y1-y2)/float(scale)),(float(z1-z2)/float(scale))
# def get_angel_of_unit_vectors(x1,y1,x2,y2):
def remove_begining_and_ending_of_features(labels,data):
    assert len(labels)==len(data)
    new_labels=[]
    new_data=[]
    for i in range(0,len(labels)):
        frames=data[i]
        if len(frames)==0:
            continue
        init_LHX=frames[0][LHX]
        init_LHY=frames[0][LHY]
        init_LHZ=frames[0][LHZ]
        init_RHX=frames[0][RHX]
        init_RHY=frames[0][RHY]
        init_RHZ=frames[0][RHZ]        
        new_frames=[]
        for frame in frames:
            left_hand_dis=distance(init_LHX,init_LHY,init_LHZ,frame[LHX],frame[LHY],frame[LHZ])
            right_hand_dis=distance(init_RHX,init_RHY,init_RHZ,frame[RHX],frame[RHY],frame[RHZ])
            dis=left_hand_dis+right_hand_dis
            if dis<INIT_THRESHOLD:
                continue  
            new_frames.append(frame)
        if len(new_frames)>=MIN_FRAME_NUM:
            new_labels.append(labels[i])
            new_data.append(new_frames)
    return new_labels,new_data

def construct_adhoc_features(labels,data):
    assert len(labels)==len(data)
    features=[]
    for i in range(0,len(labels)):
        frames=data[i]
        init_LHX=frames[0][LHX]
        init_LHY=frames[0][LHY]
        init_LHZ=frames[0][LHZ]
        init_RHX=frames[0][RHX]
        init_RHY=frames[0][RHY]
        init_RHZ=frames[0][RHZ]        
        fa=0
        fb=0,0
        fc=0,0
        fd=0,0
        fk=[0]*MOG_LENGTH
        fl=[0]*MOG_LENGTH
        features.append([])
#         features=[]
        for frame in frames:
            left_hand_dis=distance(init_LHX,init_LHY,init_LHZ,frame[LHX],frame[LHY],frame[LHZ])
            right_hand_dis=distance(init_RHX,init_RHY,init_RHZ,frame[RHX],frame[RHY],frame[RHZ])
            dis=left_hand_dis+right_hand_dis
#             print dis
#             if dis<INIT_THRESHOLD:
#                 continue
    
            scale=frame[RSZ]
            fa=distance(frame[LHX],frame[LHY],frame[LHZ],frame[RHX],frame[RHY],frame[RHZ])/scale#(float(frame[HIPY])-float(frame[HY]))
#             fb=get_2d_unit_vector(frame[LHX],frame[LHY],frame[LSX],frame[LSY])
#             fc=get_2d_unit_vector(frame[RHX],frame[RHY],frame[RSX],frame[RSY])
#             fd=get_2d_unit_vector(frame[RHX],frame[RHY],frame[LHX],frame[LHY])
#             fe=angle((frame[LSX]-frame[LEX],frame[LSY]-frame[LEY]),(frame[LHX]-frame[LEX],frame[LHY]-frame[LEY]))
#             ff=angle((frame[RSX]-frame[REX],frame[RSY]-frame[REY]),(frame[RHX]-frame[REX],frame[RHY]-frame[REY]))
            fb=get_unit_vector(frame[REX],frame[REY],frame[REZ],frame[RSX],frame[RSY],frame[RSZ])
            fc=get_unit_vector(frame[RHX],frame[RHY],frame[RHZ],frame[REX],frame[REY],frame[REZ])
            fd=get_unit_vector(frame[LEX],frame[LEY],frame[LEZ],frame[LSX],frame[LSY],frame[LSZ])
            fe=get_unit_vector(frame[LHX],frame[LHY],frame[LHZ],frame[LEX],frame[LEY],frame[LEZ])
            ff=get_unit_vector(frame[LHX],frame[LHY],frame[LHZ],frame[RHX],frame[RHY],frame[RHZ])
#             fb=get_vector(frame[LHX],frame[LHY],frame[LHZ],frame[LSX],frame[LSY],frame[LSZ],1)
#             fc=get_vector(frame[RHX],frame[RHY],frame[RHZ],frame[RSX],frame[RSY],frame[RSZ],1)
#             fd=get_vector(frame[RHX],frame[RHY],frame[RHZ],frame[LHX],frame[LHY],frame[LHZ],1)
            fg=angle((frame[LSX]-frame[LEX],frame[LSY]-frame[LEY],frame[LSZ]-frame[LEZ]),(frame[LHX]-frame[LEX],frame[LHY]-frame[LEY],frame[LHZ]-frame[LEZ]))
            fh=angle((frame[RSX]-frame[REX],frame[RSY]-frame[REY],frame[RSZ]-frame[REZ]),(frame[RHX]-frame[REX],frame[RHY]-frame[REY],frame[RHZ]-frame[REZ]))
            fi=angle((frame[HX]-frame[LSX],frame[HY]-frame[LSY],frame[HZ]-frame[LSZ]),(frame[LEX]-frame[LSX],frame[LEY]-frame[LSY],frame[LEZ]-frame[LSZ]))
            fj=angle((frame[HX]-frame[RSX],frame[HY]-frame[RSY],frame[HZ]-frame[RSZ]),(frame[REX]-frame[RSX],frame[REY]-frame[RSY],frame[REZ]-frame[RSZ]))
            if frame[LEFT_MOG]:
                fk=hog(frame[LEFT_MOG])
            if frame[RIGHT_MOG]:
                fl=hog(frame[RIGHT_MOG])
#             fp=distance(frame[LHX],frame[LHY],frame[LHZ],frame[LSX],frame[LSY],frame[LSZ])/scale
#             fq=distance(frame[RHX],frame[RHY],frame[RHZ],frame[RSX],frame[RSY],frame[RSZ])/scale
#             fg=frame[ALI1]
#             fh=frame[AR1]
#             fi=frame[ALI2]
#             fj=frame[AR2]
#             fk=frame[HAND_COUNT]
#             fl=frame[TIP_COUNT1]
#             fm=frame[TIP_COUNT2]
            features[i]+=[fa,fb[0],fb[1],fb[2],fc[0],fc[1],fc[2],fd[0],fd[1],fd[2],fe[0],fe[1],fe[2],ff[0],ff[1],ff[2],fg,fh,fi,fj]#+list(fk)+list(fl)
            #print '!!!',len(features[i])
#             features[i]+=
#             features.append([fa,fb[0],fb[1],fb[2],fc[0],fc[1],fc[2],fd[0],fd[1],fd[2],fe[0],fe[1],fe[2],ff[0],ff[1],ff[2],fg,fh,fi,fj])#+list(fk)+list(fl))
#             fe=get_unit_vector()
#             print features[i]
#         break
#         print features
#         ret_data.append(features)
    return labels,features

def find_handshape_indices_for_a_frame(templates,hog):
    ret=[]
    for i in range(0,len(templates)):
        if hog_distance(templates[i],hog)<TEMPLATE_THRESHOLD:
            ret.append(i)
    return ret

def construct_accumulative_features(frames):
    templates=load_templates()
    left_features=[0.0]*len(templates)
    right_features=[0.0]*len(templates)
    for frame in frames:
        if frame[LEFT_HOG]:
            left_features=[hog_distance(templates[i],hog(frame[LEFT_HOG]))+left_features[i] for i in range(0,len(templates))]
        if frame[RIGHT_HOG]:
            right_features=[hog_distance(templates[i],hog(frame[RIGHT_HOG]))+right_features[i] for i in range(0,len(templates))]
    return normalize_histogram(left_features)+normalize_histogram(right_features)

def construct_binary_features(length,indices):
    ret=[0]*length
    if len(indices)<10:
        return ret
    for i in indices:
        ret[i]+=1
    return ret
    
def construct_hog_binary_features(frames,templates,level_index,level_num=3):
    if level_index==level_num:
        return []
    left_indices=[]
    right_indices=[]
    for frame in frames:
        t1=time.time()
        if frame[LEFT_HOG]:
            left_indices+=(find_handshape_indices_for_a_frame(templates,hog(frame[LEFT_HOG])))
        if frame[RIGHT_HOG]:
            right_indices+=(find_handshape_indices_for_a_frame(templates,hog(frame[RIGHT_HOG])))
        t2=time.time()
        #print t2-t1
#     print left_indices,right_indices
    left_histogram=construct_hog_binary_features(frames[0:len(frames)/2],templates,level_index+1,level_num)
    right_histogram=construct_hog_binary_features(frames[len(frames)/2+1:len(frames)],templates,level_index+1,level_num)
    return normalize_histogram(construct_binary_features(len(templates),left_indices))+normalize_histogram(construct_binary_features(len(templates),right_indices))+left_histogram+right_histogram

def hog_distance_histogram(frames,level_num=3):
    if level_num==0:
        return []
    templates=load_templates()
    left_hand_feature=[0.0]*len(templates)
    right_hand_feature=[0.0]*len(templates)
    for frame in frames:
        for i in range(0,len(templates)):
            if frame[LEFT_HOG]:
                left_hand_feature[i]+=hog_distance(hog(frame[LEFT_HOG]),templates[i])/float(len(frames))
            if frame[RIGHT_HOG]:
                right_hand_feature[i]+=hog_distance(hog(frame[RIGHT_HOG]),templates[i])/float(len(frames))
    left_histogram=hog_distance_histogram(frames[0:len(frames)/2],level_num-1)
    right_histogram=hog_distance_histogram(frames[len(frames)/2+1:len(frame)],level_num-1)
    return normalize_histogram(left_hand_feature)+normalize_histogram(right_hand_feature)+left_histogram+right_histogram
    
def construct_features(labels,data,bin_num=8,level_num=2,level_num_hog=3):
    assert len(labels)==len(data)
    templates=load_templates("../data/hog_60template15mean.txt")
    ret_labels=[]
    features=[]
    for i in range(0,len(labels)):
        frames=data[i]      
        if len(frames)==0:
            print i,labels[i],frames
            continue
        movement_descriptor=hod(frames,bin_num,level_num)
        hand_shape_descriptor=construct_hog_binary_features(frames,templates,1,level_num_hog)
        if sum(hand_shape_descriptor)==0:
            print i,'no length'
            continue
        ret_labels.append(labels[i])
#         print "extracting feature:",i
        features.append(normalize_histogram(movement_descriptor)+normalize_histogram(hand_shape_descriptor))
#         features.append(hod(frames))
#         features.append(construct_hog_binary_features(frames,templates))#+construct_accumulative_features(frames))
#         features.append(construct_hog_binary_features(frames,templates))#+construct_accumulative_features(frames))
    return ret_labels,features

def train_svm_model(labels,data,para='-t 0 -c 1000 -b 1'):
    assert len(labels)==len(data)
    prob  = svm_problem(labels,data) 
#     param = svm_parameter('-t 0 -c 4 -b 1')
    param = svm_parameter(para)

    ## training the model
    m = svm_train(prob, param)
    return m
    #testing the model
#     x0, max_idx = gen_svm_nodearray([1, 1 , 1])
#     print libsvm.svm_predict(m, x0)
def test_svm_model(m,labels,data):
    p_labels, p_acc, p_vals = svm_predict(labels, data, m)
    return p_labels, p_acc, p_vals

def init_hmm_model(m,d):
    n=4
    pi = numpy.array([0]*n)
    pi[0]=1.0
    A = numpy.zeros((n,n),dtype=numpy.double)/float(n)
    A[0][0]=0.3
    A[0][1]=0.3
    A[0][2]=0.4    
    A[1][1]=0.3
    A[1][2]=0.3
    A[1][3]=0.4
    A[2][2]=0.5
    A[2][3]=0.5
    A[3][3]=1.0
#     A[3][4]=0.5
#     A[4][4]=1.0
    w = numpy.ones((n,m),dtype=numpy.double)/float(m)
    means = numpy.ones((n,m,d),dtype=numpy.double)
    covars = [[ numpy.matrix(numpy.eye(d,d)) for j in xrange(m)] for i in xrange(n)]
     
    for i in range(0,n):
        for j in range(0,m):
            for k in range(0,d):
                means[i][j][k]=0.5

    gmmhmm = GMHMM(n,m,d,A,means,covars,w,pi,init_type='user',verbose=True)
    return gmmhmm 

def init_hmms(m,d,classes):
    hmms={}
    for i in classes:
        hmms[i]=init_hmm_model(m,d)
    return hmms

def train_one_hmm_model(hmm,frame_features,iter=10):    
    obs = numpy.array(frame_features)
    hmm.train(obs,iter)

def convert_to_frame_grained_features(frame_features):
    features=[]
    for j in range(0,len(frame_features),FEATURE_SIZE_PER_FRAME):
        features.append(frame_features[j:j+FEATURE_SIZE_PER_FRAME])
    return features

def train_hmm_models(hmms,labels,frames_features,iter=10):
    for i in range(0,len(labels)):#for all cases
        features=shrink_features(frames_features[i])
        for f in features:
            print f
        train_one_hmm_model(hmms[labels[i]],features,iter)

def test_one_hmm_model(hmm,frame_features):
#     print frame_features
    prob=hmm.forwardbackward(shrink_features(frame_features))
    return prob

def test_hmm_models_for_one_case(hmms,classes,frame_features):
    max_prob=-1000000000
    max_label=-1
    for c in classes:# for all hmms
        prob=test_one_hmm_model(hmms[c],frame_features)
#         print 'prob',prob
        if max_prob<prob:
            max_prob=prob
            max_label=c
#             print 'max_prob',max_prob
    return max_label

def test_hmm_models(hmms,labels,frames_features):
    assert len(labels)==len(frames_features)
    ACC_count=0.0
    p_labels=[]
    for i in range(0,len(labels)):#for all cases
        predicted_label=test_hmm_models_for_one_case(hmms,set(labels),frames_features[i])
        p_labels.append(predicted_label)
        if predicted_label==labels[i]:
            ACC_count+=1
    return p_labels,float(ACC_count)/len(labels)

def convert_features(feature,frame_num=6):
    original_size=len(feature)/FEATURE_SIZE_PER_FRAME
#     int feature
#     print original_size,frame_num
    assert original_size>=frame_num
     
    scale_rate=original_size/frame_num
    ret=[]
    for i in range(0,frame_num):
        ret+=feature[i*scale_rate*FEATURE_SIZE_PER_FRAME:i*scale_rate*FEATURE_SIZE_PER_FRAME+FEATURE_SIZE_PER_FRAME]
    return ret

def shrink_features(feature,frame_num=MIN_FRAME_NUM):
    original_size=len(feature)/FEATURE_SIZE_PER_FRAME
#     int feature
#     print original_size,frame_num
    assert original_size>=frame_num
     
    scale_rate=original_size/frame_num
    ret=[]
    for i in range(0,frame_num):
        ret.append(feature[i*scale_rate*FEATURE_SIZE_PER_FRAME:i*scale_rate*FEATURE_SIZE_PER_FRAME+FEATURE_SIZE_PER_FRAME])
    return ret

def shuffle_data(labels,data):
    assert len(labels)==len(data)
    import random
    ret_l=[]
    ret_d=[]
    order=range(0,len(labels))
    random.shuffle(order)
#     print order
    for i in order:
        ret_l.append(labels[i])
        ret_d.append(data[i])
    return ret_l,ret_d

def split_data(labels,data,classes_in_test=range(1,51)):
#     labels,data=shuffle_data(labels,data)
    test_indexes=[]
#     flag=[0]*(len(set(labels))+1)
    flag=set()
    for i in range(0,len(labels)):
        if not (labels[i] in flag) and labels[i] in classes_in_test:
            test_indexes.append(i)
            flag.add(labels[i])
    train_labels=[]
    train_data=[]
    test_labels=[]
    test_data=[]
    #for i in range(0, min(len(labels),len(data))-1):
    for i in range(0, len(labels)):
        if not (labels[i] in classes_in_test):
            continue
        if i in test_indexes:
            test_labels.append(labels[i])
#             test_data.append(convert_features(data[i]))
            test_data.append(data[i])
        else:
            train_labels.append(labels[i])
#             train_data.append(convert_features(data[i]))
            train_data.append(data[i])
    return train_labels,train_data,test_labels,test_data

def get_classes_with_at_least_num_of_data(labels,num=3):
    ret=set()
    count={}
    for label in labels:
        if not count.has_key(label):
            count[label]=1
        else:
            count[label]+=1
    for label in labels:
        if count[label]>=num:
            ret.add(label)
    return ret
    
# def normalize(v):
#     numpy.sum(numpy.abs(x)**2,axis=-1)**(1./2)
def retrieve_data(file_name):
    labels=[]
    data=[]
    data_file=open(file_name,'r')
    for line in data_file:
        labels.append(float(line.split()[0]))
        data.append([float(item) for item in line.split()[1:]])
    return labels,data
    
def experiment_on_hod(labels,raw_data,bin_num=4,level_num=2,level_num_hog=3,para='-s 0 -c 2048 -t 2 -g 0.5'):
    labels,data=construct_features(labels,raw_data,bin_num,level_num,level_num_hog)
# #          
#     sample_input=open('../data/100data-{}-{}.txt'.format(bin_num,level_num),'w')
#     for i in range(0,len(labels)):
#         sample_input.write('{}\t'.format(labels[i]))
#         for fea in data[i]:
#             sample_input.write('{}\t'.format(fea))
#         sample_input.write('\n')
#     sample_input.close()
    
#     labels,data=retrieve_data('../data/100data-{}-{}.txt'.format(bin_num,level_num))
#     labels,data=shuffle_data(labels,data)

#     data=normalize(data)
#     labels,data=shuffle_data(labels,data)

    train_labels,train_data,test_labels,test_data=split_data(labels,data,get_classes_with_at_least_num_of_data(labels,num=3))

#     hmms=init_hmms(1,FEATURE_SIZE_PER_FRAME,set(labels))
#     train_hmm_models(hmms,train_labels,train_data,50)    
#     hmm_res=test_hmm_models(hmms,test_labels,test_data)
    svm_m1=train_svm_model(train_labels,train_data)
    svm_res1=test_svm_model(svm_m1,test_labels,test_data)
    pred_labels=svm_res1[0];
    pred_three=[];
    right=0;
    rightonly=0;
    list1=[];
    list2=[];
    for i in range(0,len(test_labels)):
        pred_labels[i]=int(pred_labels[i]);
        third=pred_labels[i]%1000;
        second=(pred_labels[i]-third)/1000%1000;
        first=int(pred_labels[i]/1000000);
        pred_three.append([first,second,third]);
        if test_labels[i]==first or test_labels[i]==second or test_labels[i]==third:
            right=right+1;
       # else:
       #     print classNo2Label[test_labels[i]],classNo2Label[first],classNo2Label[second],classNo2Label[third]
        if test_labels[i]==first:
            rightonly=rightonly+1;
#            list1.append(test_labels[i]);
#            list2.append(first);
#        else:
#            print classNo2Label[test_labels[i]],classNo2Label[first]
    str="top3:right=%r,total=%r,accurary=%10.3f%%"%(right,len(test_labels),100*right/len(test_labels));
    str1="top1:right=%r,total=%r,accurary=%10.3f%%"%(rightonly,len(test_labels),100*rightonly/len(test_labels));
    print str
    print str1

    return svm_res1[1][0]

def experiment_on_hmm(labels,raw_data):
    labels,data=construct_adhoc_features(labels,raw_data)
#         return
#         for x in d:
#             print 'length',len(x)
    train_labels,train_data,test_labels,test_data=split_data(labels,data,get_classes_with_at_least_num_of_data(labels,num=3))
    hmms=init_hmms(1,FEATURE_SIZE_PER_FRAME,set(labels))
    train_hmm_models(hmms,train_labels,train_data)    
    hmm_res=test_hmm_models(hmms,test_labels,test_data)
    return hmm_res[1]*100    
def test_svm(labels,data,bin_num=4,level_num=2,level_num_hog=3,para='-s 0 -c 2048 -t 2 -g 0.5'):
    train_labels,train_data,test_labels,test_data=split_data(labels,data,get_classes_with_at_least_num_of_data(labels,num=3))
    svm_m1=train_svm_model(train_labels,train_data)
    svm_res1=test_svm_model(svm_m1,test_labels,test_data)
    #plot_a_graph();
    
    pred_labels=svm_res1[0];
    pred_three=[];
    right=0;
    rightonly=0;
    list1=[];
    list2=[];
    for i in range(0,len(test_labels)):
        pred_labels[i]=int(pred_labels[i]);
        third=pred_labels[i]%1000;
        second=(pred_labels[i]-third)/1000%1000;
        first=int(pred_labels[i]/1000000);
        pred_three.append([first,second,third]);
        if test_labels[i]==first or test_labels[i]==second or test_labels[i]==third:
            right=right+1;
        if test_labels[i]==first:
            rightonly=rightonly+1;
#            list1.append(test_labels[i]);
#            list2.append(first);
#        else:
#            print test_labels[i],first
            #print classNo2Label[test_labels[i]],classNo2Label[first]
    str2="top3:right=%r,total=%r,accurary=%10.3f%%"%(right,len(test_labels),100*right/len(test_labels));
    str1="top1:right=%r,total=%r,accurary=%10.3f%%"%(rightonly,len(test_labels),100*rightonly/len(test_labels));
    print str2
    print str1
    return 0;


if __name__ == '__main__':
    level1=2;
    level2=2;
#    rawname='database_empty';
#    rawname2='database_empty.db';
#    rawname='Aaron_91_140';
#    rawname2='Aaron 91-140.db';
    rawname='Aaron_141_181';
    rawname2='Aaron 141-181.db';
#    rawname='database';
#    rawname2='database.db';
    tablename=rawname+'_%r_%r'%(level1-1,level2-1);
#     labels,raw_data=[],[]
    strname='../data/'+tablename;
    databasename='../data/'+rawname2;
    labels,data,classNo2Label=load_data_no_mog(databasename);
#     labels,raw_data=remove_begining_and_ending_of_features(labels,raw_data)
#     res=experiment_on_hmm(labels,raw_data)
#     print res

#     for bn in range(4,16,4):
#         for ln in range(1,5):
#             rate=0.0
#             for i in range(0,20):
#                 rate+=experiment_on_hod(labels,raw_data,bin_num=bn,level_num=ln)
#             rate/=20.0
#             res.append((bn,ln,rate))
    res_file=open('results.txt','w')
    TEMPLATE_THRESHOLD=0.18
    db_file_name="../data/features.db";
    db2 = sqlite3.connect(db_file_name);
    cu=db2.cursor(); 
#    try:
#    cu.execute("select label,data from "+tablename);
    try:
        cu.execute("select label,data from "+tablename);
        p=cu.fetchall();
        labels=[];
        data=[];
        for i in range(len(p)):
            ptemp=p[i][1].replace('\\n','\n');
            ptemp=str(ptemp);
            t2 = pickle.loads(ptemp);
            labels.append(p[i][0]);
            data.append(t2);                   
        test_svm(labels,data,bin_num=8,level_num=level1,level_num_hog=level2);
    except:
        labels,data=construct_features(labels,data,bin_num=8,level_num=level1,level_num_hog=level2)
        db = sqlite3.connect(db_file_name);
        cu=db.cursor()  
        strname="create table "+tablename+"(id integer primary key,label integer,data blob)"
        db.execute(strname);
        for i in range (0,len(labels)):
            p1 = pickle.dumps(data[i]);
            strx = "insert into "+tablename+" values(%r,%r,%r)"%(i,labels[i],p1);
            cu.execute(strx);
        db.commit();

#        labels,data=construct_features(labels,data,bin_num=8,level_num=level1,level_num_hog=level2)
#        db = sqlite3.connect(db_file_name);
#        cu=db.cursor()  
#        strname="create table "+tablename+"(id integer primary key,label integer,data blob)"
#        db.execute(strname);
#        for i in range (0,len(labels)):
#            p1 = pickle.dumps(data[i]);
#            strx = "insert into "+tablename+" values(%r,%r,%r)"%(i,labels[i],p1);
#            cu.execute(strx);
#        db.commit();

    #res_file.write('{}\t{}\t{}\n'.format(TEMPLATE_THRESHOLD,rate,len(labels)))
    #res_file.flush()
#    for thre in range(0,50):
 #       rate=experiment_on_hod(labels,raw_data,bin_num=8,level_num=2)
#        res_file.write('{}\t{}\t{}\n'.format(TEMPLATE_THRESHOLD,rate,len(labels)))
#        res_file.flush()
#        TEMPLATE_THRESHOLD+=0.005
    #res_file.close()
#     for r in res:
#         print r[0],r[1]
        
#     f1=open('../data/single-data.txt')
#     f2=open('../data/single-data-svm.txt','w')
#     for line in f1:
#         f2.write('{}\t'.format(line.split()[0]))
#         i=0
#         for item in line.split()[1:]:
#             f2.write('{}:{}\t'.format(i,item))
#             i+=1
#         f2.write('\n')    
#     f2.close()
#     templates=load_templates()
#     handshapes=load_handshapes()    
# #     print len(handshapes)   
#     for shape in handshapes:
#         s=''
#         for template in templates:
#             s+='%f\t'%hog_distance(template,shape)
#         print s
