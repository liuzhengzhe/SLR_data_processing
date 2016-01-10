import sys
sys.path.append('/home/lzz/caffe/caffe-master/python/')
import caffe
class caffeDL():
    def __init__(self,proto,model):
        #caffe_root ="/home/lzz/caffe-master/"
        self.net = caffe.Classifier(proto, model)
        #self.net = caffe.Classifier(caffe_root + 'new/proto/lenet_test.prototxt', caffe_root + 'model/del_inter_3/lenet__iter_11900.caffemodel')
        #self.net = caffe.Classifier('/home/lzz/caffe-master/model/bvlc_reference_caffenet/deploy.prototxt', '/home/lzz/caffe-master/model/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')

        #self.net = caffe.Classifier(caffe_root + 'new0/proto/lenet_test.prototxt', caffe_root + 'model/44_class/lenet__iter_8000.caffemodel')
        self.net.set_phase_test()
        self.net.set_mode_cpu()
    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    #net.set_mean('data', np.load(caffe_root + 'mean.binaryproto'))  # ImageNet mean
        self.net.set_raw_scale('data', 1)  # the reference model operates on images in [0,255] range instead of [0,1]
        self.net.set_channel_swap('data', (2,1,0))

