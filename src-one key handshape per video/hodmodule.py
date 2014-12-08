'''
Created on 2014-9-16

@author: lenovo
'''
from basic import *
from constant_numbers import *
from plot_graph import *
import math
def get_bin_for_vec(vec_2d,bin_num):
    if vec_2d==(0,0):
        return 0
    ang=angle(vec_2d,(1,0))
    if vec_2d[1]<0:
        ang=2*math.pi-ang
    ret= int(ang/(2*math.pi)*bin_num)
    return ret

def hod(features,bin_num=8,level_num=4):
    his=[]
    left=hod_of_a_joint(features,LHX,LHY,LHZ,bin_num,level_num)
    his+=list(left)[0]
    deter_left=left[1]
    his+=hod_of_a_joint_origin(features,LSX,LSY,LSZ,bin_num,level_num)
    his+=hod_of_a_joint_origin(features,LEX,LEY,LEZ,bin_num,level_num)
    right=hod_of_a_joint(features,RHX,RHY,RHZ,bin_num,level_num)
    his+=list(right)[0]
    deter_right=right[1]
    
    his+=hod_of_a_joint_origin(features,RSX,RSY,RSZ,bin_num,level_num)
    his+=hod_of_a_joint_origin(features,REX,REY,REZ,bin_num,level_num)
    his+=hod_of_a_joint_origin(features,LWX,LWY,LWZ,bin_num,level_num)
    his+=hod_of_a_joint_origin(features,RWX,RWY,RWZ,bin_num,level_num)

#     print len(his),his
    return his,deter_left,deter_right

def hod_of_a_joint_origin(features,x_index,y_index,z_index,bin_num,level_num):  
    his=get_hod_for_a_node_of_pyramid_origin(features,x_index,y_index,z_index,bin_num,1,level_num)              
    return his

def hod_of_a_joint(features,x_index,y_index,z_index,bin_num,level_num):  
    ret=get_hod_for_a_node_of_pyramid(features,x_index,y_index,z_index,bin_num,1,level_num) 
    his=ret[0]
    determine=ret[1]             
    return his,determine


def get_hod_for_a_node_of_pyramid_origin(features,x_index,y_index,z_index,bin_num,level_index,level_num):
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
        
        
    left_hod=get_hod_for_a_node_of_pyramid_origin(features[0:len(features)/2],x_index,y_index,z_index,bin_num,level_index+1,level_num)
    right_hod=get_hod_for_a_node_of_pyramid_origin(features[len(features)/2+1:len(features)],x_index,y_index,z_index,bin_num,level_index+1,level_num)
    bin_xy=normalize_histogram(bin_xy)
    bin_xz=normalize_histogram(bin_xz)
    bin_yz=normalize_histogram(bin_yz)
    return bin_xy+bin_xz+bin_yz+left_hod+right_hod    
def get_hod_for_a_node_of_pyramid(features,x_index,y_index,z_index,bin_num,level_index,level_num):
    if level_index==level_num:
        return []
    length_list=[]
    px=features[0][x_index]
    py=features[0][y_index]
    pz=features[0][z_index]
    bin_xy=[0]*bin_num
    bin_xz=[0]*bin_num
    bin_yz=[0]*bin_num
    flag=0
    y0=py
    determine=[0]*len(features)
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
        
        len_of_v=math.sqrt((x-px)**2+(y-py)**2+(z-pz)**2);
        length_list.append(len_of_v)
        if(flag==0 and len_of_v<0.1 and y-y0>0.2):
            flag=1;
        elif(flag>0 and len_of_v<0.1 and y-y0>0.2):
            flag+=1
        elif(len_of_v>=0.1):
            flag=0
        
        if(flag==3):
            flag=0
            for j in range(3):
                determine[i-j]=1
#                print i-j
        
        
#         print bin_xz
        px=x
        py=y
        pz=z
    
    
    
#    index=[]  
    
#    for i in range(len(features)):
#        index.append(i)
        
#    plot_curve(index,length_list)
    
        
    left_hod=get_hod_for_a_node_of_pyramid_origin(features[0:len(features)/2],x_index,y_index,z_index,bin_num,level_index+1,level_num)
    right_hod=get_hod_for_a_node_of_pyramid_origin(features[len(features)/2+1:len(features)],x_index,y_index,z_index,bin_num,level_index+1,level_num)
    bin_xy=normalize_histogram(bin_xy)
    bin_xz=normalize_histogram(bin_xz)
    bin_yz=normalize_histogram(bin_yz)
    return bin_xy+bin_xz+bin_yz+left_hod+right_hod,determine
'''fix the threshold to determine key handshape
def get_hod_for_a_node_of_pyramid(features,x_index,y_index,z_index,bin_num,level_index,level_num):
    if level_index==level_num:
        return []
    length_list=[]
    px=features[0][x_index]
    py=features[0][y_index]
    pz=features[0][z_index]
    bin_xy=[0]*bin_num
    bin_xz=[0]*bin_num
    bin_yz=[0]*bin_num
    flag=0
    y0=py
    determine=[0]*len(features)
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
        #len_of_v=y;
        len_of_v=math.sqrt((x-px)**2+(y-py)**2+(z-pz)**2);
        length_list.append(len_of_v)
        if(flag==0 and len_of_v<0.1 and y-y0>0.2):
            flag=1;
        elif(flag>0 and len_of_v<0.1 and y-y0>0.2):
            flag+=1
        elif(len_of_v>=0.1):
            flag=0
        
        if(flag==3):
            flag=0
            for j in range(3):
                determine[i-j]=1
#                print i-j
        
        
#         print bin_xz
        px=x
        py=y
        pz=z
    
    
    
    index=[]  
    
    for i in range(len(features)):
        index.append(i)
        
#    plot_curve(index,length_list)
    
        
    left_hod=get_hod_for_a_node_of_pyramid_origin(features[0:len(features)/2],x_index,y_index,z_index,bin_num,level_index+1,level_num)
    right_hod=get_hod_for_a_node_of_pyramid_origin(features[len(features)/2+1:len(features)],x_index,y_index,z_index,bin_num,level_index+1,level_num)
    bin_xy=normalize_histogram(bin_xy)
    bin_xz=normalize_histogram(bin_xz)
    bin_yz=normalize_histogram(bin_yz)
    return bin_xy+bin_xz+bin_yz+left_hod+right_hod,determine'''
'''original copy
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
'''