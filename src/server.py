import socket
import base64
import cv2
from json import JSONEncoder,JSONDecoder
import json
import numpy as np
import sys
import caffe
import datetime
address=("192.168.212.96",8080)
socket=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
socket.bind(address)
socket.listen(1)
caffe_root ="/home/lzz/caffe-master/"  # this file is expected to be i {caffe_root}/examples

sys.path.insert(0, caffe_root + 'python') 

net = caffe.Classifier(caffe_root + 'new/proto/lenet_test.prototxt',
                               caffe_root + 'lenet_iter_3500.caffemodel')
net.set_phase_test()
net.set_mode_cpu()
# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
#net.set_mean('data', np.load(caffe_root + 'mean.binaryproto'))  # ImageNet mean
net.set_raw_scale('data', 1)  # the reference model operates on images in [0,255] range instead of [0,1]
net.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB]
while(1):
   # try:
        #connection.settimeout(5) 
        connection,address=socket.accept()
        data=connection.recv(5120000)
        #print data
        decodedDict = JSONDecoder().decode(data)
        #decodedDict = json.loads(data)
        encoded_image =decodedDict ['image']
        image_shape = decodedDict ['shape']
        depthFrame = base64.decodestring(encoded_image)
        depthFrame = np.frombuffer(depthFrame,dtype='uint8')
        img=depthFrame.reshape(image_shape)

    
        #net.predict([caffe.io.load_image(imgpath)])
        net.predict([img])
        feat = net.blobs['prob'].data[4].flatten().tolist()
        #print feat
        ind=feat.index(max(feat))
        #print ind
        print datetime.datetime.now()
        '''tmpS = ''
        
        for i,f in enumerate(feat):
           tmpS += str(f) + ' '
    
        f = file('handshape.txt', 'a')
        f.write(tmpS)
        f.close()'''
        
        
        
        
        
        
        
        
        
    #    connection.send('bye')
        connection.close()
    #except:
     #   print "timeout"
socket.close()
'''

if __name__ == '__main__':  
        import socket  
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  
        sock.bind(('localhost', 8001))  
        sock.listen(5)  
        while True:  
            connection,address = sock.accept()  
            try:  
                connection.settimeout(5)  
                buf = connection.recv(1024)  
                if buf == '1':  
                    connection.send('welcome to server!')  
                else:  
                    connection.send('please go out!')  
            except socket.timeout:  
                print 'time out'  
            connection.close()  '''

