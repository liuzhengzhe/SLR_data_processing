'''import cv2
im = cv2.imread("/home/lzz/1.jpg",cv2.CV_LOAD_IMAGE_COLOR)
#im = cv2.cv.LoadImage("/home/lzz/1.jpg",cv2.CV_LOAD_IMAGE_COLOR)
cv2.imshow("1",im)
cv2.waitKey()'''

import caller

if __name__=="__main__":
    def callback(respose_data):
        print respose_data
    caller.ds_asyncore(('192.168.212.96', 8080),callback,timeout=5)
'''def my_callback(input):
    print "function my_callback was called with %s input" % (input,)
 
def caller(input, func):
    func(input)
 
for i in range(5):
    caller(i, my_callback)'''