'''
Created on 2014-9-16

@author: lenovo
'''

from basic import *
import time
from constant_numbers import *
import struct
from load import *
'''
def find_handshape_indices_for_a_frame(templates,hog,listx):
    ret=[]
   # f=open("hogright.txt",'w')

    for y in range(0,len(templates)):
#        print hog_distance(templates[i],hog);
        if hog_distance(templates[y],hog)<TEMPLATE_THRESHOLD:
      #  if hog_distance(templates[y],hog)<TEMPLATE_THRESHOLD and listx[y]==1:
            #print hog_distance(templates[y],hog)
            cell_mat=[[[] for col in range(11)] for row in range(11)]
            for j in range(121):
                r=0;
                for q in range(9):
                    r+=(templates[y][j*36+q]-hog[j*36+q])**2
#                print "r",r;
                
                cell_mat[j%11][int(j/11)].append(r)
                r=0
                for q in range(9,18):
                    r+=(templates[y][j*36+q]-hog[j*36+q])**2
#                print "r",r;
                cell_mat[j%11][int(j/11)].append(r)
                r=0
                for q in range(18,27):
                    r+=(templates[y][j*36+q]-hog[j*36+q])**2
#                print "r",r;
                cell_mat[j%11][int(j/11)].append(r)
                r=0
                for q in range(27,36):
                    r+=(templates[y][j*36+q]-hog[j*36+q])**2
#                print "r",r;
                cell_mat[j%11][int(j/11)].append(r)

            block_mat=[[0 for col in range(12)] for row in range(12)]
            weight=[[0 for col in range(12)] for row in range(12)]
        
            for i in range(11):
                for j in range(11):
                    if(i==2 and j==5):
                        i=2
                    block_mat[i][j]+=cell_mat[i][j][0];
                    block_mat[i+1][j]+=cell_mat[i][j][1];
                    block_mat[i][j+1]+=cell_mat[i][j][2];
                    block_mat[i+1][j+1]+=cell_mat[i][j][3];
                    weight[i][j]=weight[i][j]+1;
                    weight[i][j+1]+=1;
                    weight[i+1][j]+=1;
                    weight[i+1][j+1]+=1;
            for i in range(12):
                for j in range(12):
                    block_mat[i][j]/=weight[i][j]

            block_mat2=[[0 for col in range(12)] for row in range(12)]
            tmp=0
            for i in range(12):
                for j in range(12):
                    tmp+=block_mat[i][j]
                    block_mat2[i][j]=tmp/(12*i+j+1)
                    
            
            fdif=open("F:/study/save/generating/difference.txt","w");
            for i in range(12):
                for j in range(12):
                    fdif.write(str(block_mat[i][j])+" ")
                    #print block_mat[i][j]
            fdif.flush();
            fdif.close();
            fx=open("C:/Users/lenovo/Desktop/test.txt","w");
            fx.write(str(templates[y]).replace("["," ").replace("]"," ").replace(","," "));
            fx.write(str(list(hog)).replace("["," ").replace("]"," ").replace(","," "));
            fx.flush();
            fx.close();
            ret.append(y)
 #           print i;
#            print hog_distance(templates[i],hog);
   # f.close()
    return ret
'''

def find_handshape_indices_for_a_frame(templates,hog,list):
    ret=[]
    for i in range(0,len(templates)):
        if hog_distance(templates[i],hog)<TEMPLATE_THRESHOLD:
            ret.append(i)
    return ret




def hog_distance(h1,h2):
    assert len(h1)==len(h2)
    ret=0
    for i in range(0,len(h1)):
        ret+=((h1[i]-h2[i])**2)
        #ret+=abs(h1[i]-h2[i])
    ret=math.sqrt(ret/len(h1))
    return ret

def hog(hand_data):
    new_hog=tuple( [struct.unpack('f',hand_data[i:i+4])[0] for i in range(0,len(hand_data),4)])
#     print new_hog
    return new_hog





def construct_hog_binary_features(frames,templates,level_index,level_num,fout,f,f_length,left,right):
    if level_index==level_num:
        return []
    left_indices=[]
    right_indices=[]
    index=0
    exist=0
    f_one_video=open("../save/generating/hog/onevideo.txt","w")
    for frame in frames:
        #f=open("hogright.txt",'w')
        t1=time.time()
        if frame[LEFT_HOG]:
            left_indices+=(find_handshape_indices_for_a_frame(templates,hog(frame[LEFT_HOG]),left))
            hoglist=list(hog(frame[LEFT_HOG]));
            hog1=str(hoglist)
            hog1=str(hoglist).replace("["," ")
            hog1=hog1.replace("]"," ")
            hog1=hog1.replace(","," ")
#            if(index>0):
#                if(left[index]==1):
#                    f.write(hog1)
#                    f.flush()
        if frame[RIGHT_HOG]:
#            if(index==1):
#                print index
            right_indices+=(find_handshape_indices_for_a_frame(templates,hog(frame[RIGHT_HOG]),right))
            hoglist=list(hog(frame[RIGHT_HOG]));
            hog1=str(hoglist).replace("["," ")
            hog1=hog1.replace("]"," ")

#            if(index>0):
            if(right[index]==1):
                f.write(hog1.replace(","," "))
                f_one_video.write(hog1.replace(","," "))
                exist=exist+1
        f.flush()
        t2=time.time()
        index=index+1
        #f.close()
        #print t2-t1
#     print left_indices,right_indices
#    f.close()
    f_one_video.close()
    f_length.write(str(exist)+" ");
    f_length.flush()
    left_histogram=construct_hog_binary_features(frames[0:len(frames)/2],templates,level_index+1,level_num,fout,f,f_length,left,right)
    right_histogram=construct_hog_binary_features(frames[len(frames)/2+1:len(frames)],templates,level_index+1,level_num,fout,f,f_length,left,right)
#    index=2;
    x1=construct_binary_features(len(templates),left_indices)
    x2=construct_binary_features(len(templates),right_indices)
    
    print >>fout,sum(x1),sum(x2)

    return normalize_histogram(x1)+normalize_histogram(x2)+left_histogram+right_histogram



def construct_hog_binary_features2(frames,templates,level_index,level_num,fout):
    if level_index==level_num:
        return []
    left_indices=[]
    right_indices=[]
    for frame in frames:
        #f=open("hogright.txt",'w')
        t1=time.time()
        if frame[LEFT_HOG]:
            left_indices+=(find_handshape_indices_for_a_frame(templates,hog(frame[LEFT_HOG])))
        if frame[RIGHT_HOG]:
            right_indices+=(find_handshape_indices_for_a_frame(templates,hog(frame[RIGHT_HOG])))
        t2=time.time()
        #f.close()
        #print t2-t1
#     print left_indices,right_indices
    left_histogram=construct_hog_binary_features(frames[0:len(frames)/2],templates,level_index+1,level_num,fout)
    right_histogram=construct_hog_binary_features(frames[len(frames)/2+1:len(frames)],templates,level_index+1,level_num,fout)
#    index=2;
    x1=construct_binary_features(len(templates),left_indices)
    x2=construct_binary_features(len(templates),right_indices)
    
    #print >>fout,sum(x1),sum(x2)

    return normalize_histogram(x1)+normalize_histogram(x2)+left_histogram+right_histogram

def construct_binary_features(length,indices):
    ret=[0]*length
    if len(indices)<10:
        return ret
    for i in indices:
        ret[i]+=1
    return ret


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





















    
