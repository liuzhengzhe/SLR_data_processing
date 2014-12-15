def caffeValue(rootpath):
    import numpy as np
    import matplotlib.pyplot as plt
    import caffe
    import os
    import cv2
    # Make sure that caffe is on the python path:
    caffe_root = rootpath  # this file is expected to be i {caffe_root}/examples
    import sys
    files=os.listdir("/home/lzz/caffe-master/new/handshapes/train/13/")  
    sys.path.insert(0, caffe_root + 'python') 
    net = caffe.Classifier(caffe_root + 'new/proto/lenet_test.prototxt',
                                   caffe_root + 'lenet_iter_3500.caffemodel')
    net.set_phase_test()
    net.set_mode_gpu()
    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    #net.set_mean('data', np.load(caffe_root + 'mean.binaryproto'))  # ImageNet mean
    net.set_raw_scale('data', 1)  # the reference model operates on images in [0,255] range instead of [0,1]
    net.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB]
    #m=[caffe.io.load_image(imgpath)]
    for f in files:
        im = cv2.imread("/home/lzz/caffe-master/new/handshapes/train/13/"+f,cv2.CV_LOAD_IMAGE_COLOR)
        im=im/255.0
        net.predict([im])
        #cv2.imshow("1",im)
        #cv2.waitKey()
        feat = net.blobs['prob'].data[0].flatten().tolist()
        ind=feat.index(max(feat))
        print feat
        print str(ind)

caffeValue('/home/lzz/caffe-master/')
#cafferoot,img_path




