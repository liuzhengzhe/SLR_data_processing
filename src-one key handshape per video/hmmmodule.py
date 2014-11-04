'''
Created on 2014-9-16

@author: lenovo
'''
from constant_numbers import *
from hmm.continuous.GMHMM import GMHMM
import numpy
import basic
from numbers import *
from hogmodule import *
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
            left_hand_dis=basic.distance(init_LHX,init_LHY,init_LHZ,frame[LHX],frame[LHY],frame[LHZ])
            right_hand_dis=basic.distance(init_RHX,init_RHY,init_RHZ,frame[RHX],frame[RHY],frame[RHZ])
            dis=left_hand_dis+right_hand_dis
#             print dis
#             if dis<INIT_THRESHOLD:
#                 continue
    
            scale=frame[RSZ]
            fa=basic.distance(frame[LHX],frame[LHY],frame[LHZ],frame[RHX],frame[RHY],frame[RHZ])/scale#(float(frame[HIPY])-float(frame[HY]))
#             fb=get_2d_unit_vector(frame[LHX],frame[LHY],frame[LSX],frame[LSY])
#             fc=get_2d_unit_vector(frame[RHX],frame[RHY],frame[RSX],frame[RSY])
#             fd=get_2d_unit_vector(frame[RHX],frame[RHY],frame[LHX],frame[LHY])
#             fe=angle((frame[LSX]-frame[LEX],frame[LSY]-frame[LEY]),(frame[LHX]-frame[LEX],frame[LHY]-frame[LEY]))
#             ff=angle((frame[RSX]-frame[REX],frame[RSY]-frame[REY]),(frame[RHX]-frame[REX],frame[RHY]-frame[REY]))
            fb=basic.get_unit_vector(frame[REX],frame[REY],frame[REZ],frame[RSX],frame[RSY],frame[RSZ])
            fc=basic.get_unit_vector(frame[RHX],frame[RHY],frame[RHZ],frame[REX],frame[REY],frame[REZ])
            fd=basic.get_unit_vector(frame[LEX],frame[LEY],frame[LEZ],frame[LSX],frame[LSY],frame[LSZ])
            fe=basic.get_unit_vector(frame[LHX],frame[LHY],frame[LHZ],frame[LEX],frame[LEY],frame[LEZ])
            ff=basic.get_unit_vector(frame[LHX],frame[LHY],frame[LHZ],frame[RHX],frame[RHY],frame[RHZ])
#             fb=get_vector(frame[LHX],frame[LHY],frame[LHZ],frame[LSX],frame[LSY],frame[LSZ],1)
#             fc=get_vector(frame[RHX],frame[RHY],frame[RHZ],frame[RSX],frame[RSY],frame[RSZ],1)
#             fd=get_vector(frame[RHX],frame[RHY],frame[RHZ],frame[LHX],frame[LHY],frame[LHZ],1)
            fg=basic.angle((frame[LSX]-frame[LEX],frame[LSY]-frame[LEY],frame[LSZ]-frame[LEZ]),(frame[LHX]-frame[LEX],frame[LHY]-frame[LEY],frame[LHZ]-frame[LEZ]))
            fh=basic.angle((frame[RSX]-frame[REX],frame[RSY]-frame[REY],frame[RSZ]-frame[REZ]),(frame[RHX]-frame[REX],frame[RHY]-frame[REY],frame[RHZ]-frame[REZ]))
            fi=basic.angle((frame[HX]-frame[LSX],frame[HY]-frame[LSY],frame[HZ]-frame[LSZ]),(frame[LEX]-frame[LSX],frame[LEY]-frame[LSY],frame[LEZ]-frame[LSZ]))
            fj=basic.angle((frame[HX]-frame[RSX],frame[HY]-frame[RSY],frame[HZ]-frame[RSZ]),(frame[REX]-frame[RSX],frame[REY]-frame[RSY],frame[REZ]-frame[RSZ]))
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

def experiment_on_hmm(labels,raw_data):
    labels,data=construct_adhoc_features(labels,raw_data)
#         return
#         for x in d:
#             print 'length',len(x)
    train_labels,train_data,test_labels,test_data=basic.split_data(labels,data,get_classes_with_at_least_num_of_data(labels,num=3))
    hmms=init_hmms(1,FEATURE_SIZE_PER_FRAME,set(labels))
    train_hmm_models(hmms,train_labels,train_data)    
    hmm_res=test_hmm_models(hmms,test_labels,test_data)
    return hmm_res[1]*100   