'''
Created on Oct 13, 2014

@author: liuzz
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



if __name__ == '__main__':
    level1=2;
    level2=2;
#    rawname='database_empty';
#    rawname2='database_empty.db';
    #rawname='databasemany';
    #rawname2='databasemany.db';
#    rawname='Aaron_51_90';
#    rawname2='Aaron 51-90.db';
#    rawname='Aaron_141_181';
#    rawname2='Aaron 141-181.db';
    #rawname='database';
    #rawname2='database.db';
    rawname='database_791_821';
    rawname2='database791-821.db';
    tablename=rawname+'_%r_%r'%(level1-1,level2-1);
#     labels,raw_data=[],[]
    strname='../data/'+tablename;
    databasename='../data/'+rawname2;
#    labels,data,classNo2Label=load_data_no_mog(databasename);

    res_file=open('results.txt','w')

    db_file_name="../data/features.db";
    db2 = sqlite3.connect(db_file_name);
    cu=db2.cursor(); 
    
    key_data=open("D:/eclipse/project/save/generating/handshape/tmp.txt")
    cu.execute("select label,data from "+tablename);
    p=cu.fetchall();
    labels=[];
    data=[];
    kh2=[]
    kh=key_data.read().split()
    for i in range(len(kh)):
        kh2.append(float(kh[i]))
    
    for i in range(len(p)):
        ptemp=p[i][1].replace('\\n','\n');
        ptemp=str(ptemp);
        t2 = pickle.loads(ptemp);
        labels.append(p[i][0]);
        t2=t2[1:192]
        t3=[]
        
        for j in range(0,4356):
            
            t3.append(kh2[j+4356*i])
        t4=normalize_histogram(t3)
        t1=t2+t4
        
        
        
        
        
        
        data.append(t1);   
        
        
test_svm(labels,data,bin_num=8,level_num=level1,level_num_hog=level2);

