import sys
sys.path.append('/media/lzz/65c50da0-a3a2-4117-8a72-7b37fd81b574/sign/caffe/python/')
import numpy as np
import matplotlib.pyplot as plt
import caffe
import os
import cv2
def caffeValue(rootpath):
    net=caffe.Classifier('/home/lzz/caffe/caffe-master/examples/imagenet/train_val_16_py.prototxt','/home/lzz/caffe/caffe-master/examples/imagenet/model/4096_iter_10000.caffemodel')
    net.set_phase_test()
    net.set_mode_cpu()
    net.set_raw_scale('data', 255)

    '''net2=caffe.Classifier('/media/lzz/65c50da0-a3a2-4117-8a72-7b37fd81b574/sign/model/train_val_py_images.prototxt','/media/lzz/65c50da0-a3a2-4117-8a72-7b37fd81b574/sign/model/lzz_AM_single_iter_7000.caffemodel')
    net2.set_phase_test()
    net2.set_mode_cpu()
    net2.set_raw_scale('images', 255)

    net3=caffe.Classifier('/media/lzz/65c50da0-a3a2-4117-8a72-7b37fd81b574/sign/model/train_val_py_concate.prototxt','/media/lzz/65c50da0-a3a2-4117-8a72-7b37fd81b574/sign/model/lzz_AM_single_iter_7000.caffemodel')
    net3.set_phase_test()
    net3.set_mode_cpu()
    net3.set_raw_scale('data', 255)'''

    imgs=[]
    im = cv2.imread("/media/lzz/65c50da0-a3a2-4117-8a72-7b37fd81b574/sign/handshapes/handshapes/train/0/2_Intersect_C.jpg")
    im=im/255.0
    im=cv2.resize(im,(227,227))
    bigim=cv2.merge((im,im,im,im,im,im,im,im,im,im),2)
    print bigim.shape
    imgs.append(im)

    #np.set_printoptions(threshold='nan')

    net.predict(imgs,False)
    feature2=net.blobs['fc7'].data[0]

    f=open('/home/lzz/feature.txt','w')
    f.write(str(net.blobs['data'].data[0]))
    f.close()
    print net.blobs['fc7'].data[0]
    #print np.amax(net.blobs['data'].data[0])
    index=[]
    for i in range(500):
        index.append(i)
    '''for i in range(4):
        
        feat = net.blobs['fc7'].data[i].flatten().tolist()
        #ind=feat.index(max(feat))
        print len(feat)'''


caffeValue('/home/lzz/caffe-master/')
#cafferoot,img_path




