import random
import os

from numpy import *

if __name__ == '__main__':
    path="/media/lzz/65c50da0-a3a2-4117-8a72-7b37fd81b574/sign/handshapes/inter/"
    f1=open(path+"images.txt","w")

    for ran in range (0,24):
        for p in os.listdir(path+str(ran)):
            f1.write("/media/lzz/65c50da0-a3a2-4117-8a72-7b37fd81b574/sign/handshapes/inter/"+str(ran)+"/"+str(p)+" "+str(ran)+"\n")
