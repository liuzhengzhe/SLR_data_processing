import sys
sys.path.append('/home/lzz/caffe/caffe-master/python')
def caffeValue(rootpath):
    import numpy as np
    import matplotlib.pyplot as plt
    import caffe
    import os
    import cv2
    # Make sure that caffe is on the python path:
    caffe_root = rootpath  # this file is expected to be i {caffe_root}/examples
    import sys
    #files=os.listdir("/home/lzz/caffe-master/new/handshapes/train/13/")
    net=caffe.Classifier('/media/lzz/65c50da0-a3a2-4117-8a72-7b37fd81b574/sign/proto_hdf5/train_val_16_py.prototxt','/home/lzz/caffe/caffe-master/examples/imagenet/model/4096_iter_10000.caffemodel')
    #net = caffe.Classifier('/home/lzz/caffe-master/deeplab/test.prototxt',  '/home/lzz/caffe-master/deeplab/model.caffemodel')
    net.set_phase_test()
    net.set_mode_cpu()
    net.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    #net.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB]
    #m=[caffe.io.load_image(imgpath)]
    imgs=[]
    #for f in files:
    #im=caffe.io.load_image("/media/lzz/65c50da0-a3a2-4117-8a72-7b37fd81b574/sign/handshapes/hand/handshape/lzz+HKG_007_a_0021+18586_Right_C#.jpg")
    im = cv2.imread("/media/lzz/65c50da0-a3a2-4117-8a72-7b37fd81b574/sign/handshapes/hand/handshape/lzz+HKG_007_a_0021+18586_Right_C#.jpg")
    im=im/255.0
    im=cv2.resize(im,(227,227))

    imgs.append(im)
    np.set_printoptions(threshold='nan')

    net.predict(imgs,False)
        #print im
        #cv2.imshow("1",im)
        #cv2.waitKey()
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




