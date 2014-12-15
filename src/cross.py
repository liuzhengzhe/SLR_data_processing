'''
Created on Nov 17, 2014

@author: liuzz
'''
from numpy import *
if __name__ == '__main__':
    crossp="D:/caffe/caffe-windows/caffe-windows/examples/mnist/handshapes/"
    f1=open(crossp+"trainold.txt","r")
    f2=open(crossp+"testold.txt","r")
    train=open(crossp+"train.txt","w")
    test=open(crossp+"test.txt","w")
    datatrain=f1.readlines()
    datatest=f2.readlines()
    while(datatrain!=[]):
        t=random.randint(0,len(datatrain))
        train.write(datatrain[t])
        del datatrain[t]
    while(datatest!=[]):
        t=random.randint(0,len(datatest))
        test.write(datatest[t])
        del datatest[t]
        
        
        
        
    