import caffe
import os
import cv2
def caffeValue(rootpath):

    # Make sure that caffe is on the python path:
    caffe_root = rootpath  # this file is expected to be i {caffe_root}/examples
    import sys
    files=os.listdir("/home/lzz/caffe-master/new/handshapes/train/13/")  
    sys.path.insert(0, caffe_root + 'python') 
    net = caffe.Classifier(caffe_root + 'new/proto/lenet_test.prototxt', caffe_root + 'model/colorframe/lenet_iter_5000.caffemodel')
    net.set_phase_test()
    net.set_mode_gpu()
    net.set_raw_scale('data', 1)  # the reference model operates on images in [0,255] range instead of [0,1]
    net.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB]
    #m=[caffe.io.load_image(imgpath)]
    imgs=[]
    for f in files:
        im = cv2.imread("/home/lzz/caffe-master/4keyr.bmp")
        im=im/255.0
        imgs.append(im)

    net.predict(imgs,False)
    print len(net.blobs['prob'].data)
        #print im
        #cv2.imshow("1",im)
        #cv2.waitKey()
    '''print len(net.blobs['prob'].data)
    index=[]
    for i in range(500):
        index.append(i)
    for i in range(4):
        
        feat = net.blobs['ip1'].data[i].flatten().tolist()'''
        #ind=feat.index(max(feat))
        #print len(feat)
        #matplotlib.pyplot.plot(index,feat)
        #matplotlib.pyplot.show()

if __name__ == '__main__':
    caffeValue('/home/lzz/caffe-master/')





