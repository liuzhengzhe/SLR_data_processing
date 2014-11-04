'''
Created on Oct 14, 2014

@author: liuzz
'''
'''
Created on 2014-7-30

@author: lenovo
'''
from svmutil import *
from svmmodule import *
import matplotlib
import sqlite3
import math
import numpy
import struct
#from hmm.continuous.GMHMM import GMHMM
from cluster import KMeansClustering
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


label2classNo={}

def load_data_no_mog(db_file_name):
    RAW_DATA_LENGTH=32
    db = sqlite3.connect(db_file_name)
    cu=db.cursor()
    labels=[]
    signerlist=[]
    #label2classNo={}
    classNo2Label={}
    classNo=0
    current_data_index=-1
    current_sign_id=-1
    current_class_id=-1
    data=[]
    name=[]
    filenamelist=[]
    whole=db.execute("Select FrameData.SampleIndex,FrameData.index_ID,SignId,signer,FileName,SkeletonShoulderLeftX, \
    SkeletonShoulderLeftY,SkeletonShoulderLeftZ,SkeletonShoulderRightX,SkeletonShoulderRightY,\
    SkeletonShoulderRightZ,SkeletonElbowLeftX,SkeletonElbowLeftY,SkeletonElbowLeftZ,SkeletonElbowRightX,\
    SkeletonElbowRightY,SkeletonElbowRightZ,SkeletonHandLeftX,SkeletonHandLeftY,SkeletonHandLeftZ,\
    SkeletonHandRightX,SkeletonHandRightY,SkeletonHandRightZ,SkeletonHeadX,SkeletonHeadY,\
    SkeletonHeadZ,SkeletonHipCenterX,SkeletonHipCenterY,SkeletonHipCenterZ,SkeletonWristLeftX,\
    SkeletonWristLeftY,SkeletonWristLeftZ,SkeletonWristRightX,SkeletonWristRightY,SkeletonWristRightZ,\
    LeftHandHOG,RightHandHog from FrameData, SignSample where FrameData.SampleIndex=SignSample.index_ID\
     ORDER BY  SignSample.SignID,FrameData.SampleIndex,FrameData.FrameNumber;")
    for sign_id,frame_id,word_id,signer,filename,lsx,lsy,lsz,rsx,rsy,rsz,lex,ley,lez,rex,rey,lez,lhx,lhy,\
    lhz,rhx,rhy,rhz,hx,hy,hz,hipcx,hipcy,hipcz,lwx,lwy,lwz,rwx,rwy,rwz,leftHog,rightHog \
    in whole:    
        '''cu.execute("Select Intersected from SignSample where index_ID="+str(sign_id))
        intersected=cu.fetchall()
        inter=intersected[0][0]
        if(inter==1):
            continue'''
        if current_class_id!=word_id and (not label2classNo.has_key(word_id)):
            classNo+=1
            current_class_id=word_id
            label2classNo[word_id]=classNo
            classNo2Label[classNo]=word_id
        
        if current_sign_id!=sign_id:
            #new sign record
#            if current_data_index>=0 and len(data[current_data_index])<54:
#                print '!!!!',current_sign_id,len(data[current_data_index])
            current_sign_id=sign_id
            data.append(list())
            signerlist.append(signer)
            name.append(word_id)
            filenamelist.append(filename)
            labels.append(label2classNo[word_id]) 
            current_data_index+=1
#         print current_sign_id,len(data)
        if lsx:
            if(current_data_index<0):
                x=1
            data[current_data_index].append([lsx,lsy,lsz,rsx,rsy,rsz,lex,ley,lez,rex,rey,lez,lhx,lhy,lhz,rhx,rhy,rhz,hx,hy,hz,hipcx,hipcy,hipcz,lwx,lwy,lwz,rwx,rwy,rwz,leftHog,rightHog])
#         print data[current_data_index]
    return labels,data,name,signerlist,filenamelist,classNo2Label


def construct_features(labels,data,signer,name,bin_num=8,level_num=2,level_num_hog=3):
    assert len(labels)==len(data)
    templates=load_templates("../data/hog_60template15mean.txt")
    #templates=load_templates("F:/study/save/generating/handshape/tmp.txt")
    ret_labels=[]
    features=[]
    ret_name=[]
    ret_signer=[]
#    fout=open("test.txt","wt")
#    f=open("../save/generating/hog/tmp.txt","w")
#    f_length=open("../save/generating/kmedoid/length.txt","w")
#    for i in range(0,len(labels)):
    for i in range(len(labels)):
        frames=data[i]
        if(i==127):
            i=127      
        if len(frames)==0:
            print i,labels[i],frames
            continue
        movement=hod(frames,bin_num,level_num)
        movement_descriptor=movement[0]
        deter_left=movement[1]
        deter_right=movement[2]
    
        if(i==6):
            hand_shape_descriptor=construct_hog_binary_features(frames,templates,1,level_num_hog,deter_left,deter_right,name[i])
        else:
            hand_shape_descriptor=construct_hog_binary_features(frames,templates,1,level_num_hog,deter_left,deter_right,name[i])
        if sum(hand_shape_descriptor)==0:
            print i,'no length'
            continue
        ret_labels.append(labels[i])
        ret_name.append(name[i])
        ret_signer.append(signer[i])
#         print "extracting feature:",i
        #features.append(normalize_histogram(movement_descriptor))
        features.append(normalize_histogram(movement_descriptor)+normalize_histogram(hand_shape_descriptor))
#         features.append(hod(frames))
#         features.append(construct_hog_binary_features(frames,templates))#+construct_accumulative_features(frames))
#         features.append(construct_hog_binary_features(frames,templates))#+construct_accumulative_features(frames))
    return ret_labels,features,ret_name,ret_signer



    

 



if __name__ == '__main__':
    level1=2;
    level2=2;
    rawname='database_aaron_';
    rawname2='database_aaron.db';
#    rawname='database_empty';
#    rawname2='database_empty.db';
    #rawname='databasemany';
    #rawname2='databasemany.db';
    rawname='Aaron_1_50';
    rawname2='Aaron1-50.db';
#    rawname='Aaron_141_181';
#    rawname2='Aaron 141-181.db';
    #rawname='database';
    #rawname2='database.db';
#    rawname='database_791_821';
#    rawname2='database791-821.db';
    tablename=rawname+'_%r_%r'%(level1-1,level2-1);
#     labels,raw_data=[],[]
    strname='../data/'+tablename;
    databasename='../data/'+rawname2;
#    

    res_file=open('results.txt','w')

    db_file_name="../data/features.db";
    db2 = sqlite3.connect(db_file_name);
    cu=db2.cursor(); 
    try:
        cu.execute("select label,data from "+tablename);
        p=cu.fetchall();
        labels=[];
        data=[];
        for i in range(len(p)):
            ptemp=p[i][1].replace('\\n','\n');
            ptemp=str(ptemp);
            t2 = pickle.loads(ptemp);
#            t2=t2[0:192]
            labels.append(p[i][0]);
            data.append(t2);                
        test_svm(labels,data,bin_num=8,level_num=level1,level_num_hog=level2);
    except:
        labels,data,name,signer,filename,classNo2Label=load_data_no_mog(databasename);
        labels,data,name,signer=construct_features(labels,data,signer,filename,bin_num=8,level_num=level1,level_num_hog=level2)
        db = sqlite3.connect(db_file_name);
        cu=db.cursor()  
        strname="create table "+tablename+"(id integer primary key,label integer,data blob,name string,signer string)"
        db.execute(strname);
        for i in range (0,len(labels)):
            p1 = pickle.dumps(data[i]);
            strx = "insert into "+tablename+" values(%r,%r,%r,%r,%r)"%(i,labels[i],p1, name[i].encode("utf-8"),signer[i].encode("utf-8"));
            cu.execute(strx);
        db.commit();

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


