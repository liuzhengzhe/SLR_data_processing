import socket
import base64
import cv2
import os
from json import JSONEncoder,JSONDecoder

files=os.listdir("/home/lzz/caffe-master/new/handshapes/test/1/")  
for f in files:
    im = cv2.imread("/home/lzz/caffe-master/new/handshapes/test/1/"+f,cv2.CV_LOAD_IMAGE_COLOR)
    encoded=base64.b64encode(im)
    image_shape = im.shape
    dic={
                'image': encoded,
                'shape': image_shape
            }
    encodedJSON = JSONEncoder().encode(dic)
    #print "encoded"
    
    address=("192.168.212.96",8080)

    s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    #print "created"
    s.connect(address)
    #print "connected"
    s.send(encodedJSON)
    #s.recv(1024)
s.close()
'''
if __name__ == '__main__':  
    import socket  
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  
    sock.connect(('localhost', 8001))  
    import time  
    time.sleep(2)  
    sock.send('1')  
    print sock.recv(1024)  
    sock.close() '''
