import os
import shutil
from numpy import *
if __name__ == '__main__':


    crossp="/media/lzz/65c50da0-a3a2-4117-8a72-7b37fd81b574/sign/proto_hdf5/"


    f1=open(crossp+"traininteram","r")
    f2=open(crossp+"testinteram","r")
    train=open(crossp+"traininteram.txt","w")
    test=open(crossp+"testinteram.txt","w")
    datatrain=f1.readlines()
    datatest=f2.readlines()
    pro=crossp
    while(datatrain!=[]):
        t=random.randint(0,len(datatrain))
        train.write(datatrain[t])
        del datatrain[t]
    while(datatest!=[]):
        t=random.randint(0,len(datatest))
        test.write(datatest[t])
        del datatest[t]



