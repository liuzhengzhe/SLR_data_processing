import socket
import select
import sys

def ds_asyncore(addr,callback,timeout=5):
    s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    print addr
    s.connect(addr)
    r,w,e = select.select([s],[],[],timeout)
    if r:
        respose_data=s.recv(1024)
        callback(respose_data)
        s.close()
        return 0
    else:
        s.close()
        return 1
