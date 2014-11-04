'''
Created on 2014-9-16

@author: lenovo
'''
from constant_numbers import *
from svmmodule import *
from basic import *
from hogmodule import *
from load import *
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
            list1.append(test_labels[i]);
            list2.append(first);
        else:
            print classNo2Label[test_labels[i]],classNo2Label[first]
    str="top3:right=%r,total=%r,accurary=%10.3f%%"%(right,len(test_labels),100*right/len(test_labels));
    str1="top1:right=%r,total=%r,accurary=%10.3f%%"%(rightonly,len(test_labels),100*rightonly/len(test_labels));
    print str
    print str1

    return svm_res1[1][0]



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


def get_hop_bin_for_a_point(x1,y1,x2,y2,x,y,edge_bin_num):
    print x1,y1,x2,y2,x,y
    x_i=int((x-x1)/(x2-x1)*edge_bin_num)
    y_i=int((y-y1)/(y2-y1)*edge_bin_num)
    print "index",x_i,y_i
    return x_i,y_i
    
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
    #print left_hand_bins
    #print right_hand_bins
    return left_hand_bins+right_hand_bins

def convert_2d_to_1d(bins):
    a=numpy.array(bins)
    ret= list(a.flatten())
    return ret

    