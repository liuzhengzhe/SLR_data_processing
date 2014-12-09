import asyncore
import asynchat
import logging
import socket
import thread
class EchoServer(asyncore.dispatcher):
    def __init__(self, port):
        #print port,received_data_callback
        asyncore.dispatcher.__init__(self)
        self.create_socket(socket.AF_INET, socket.SOCK_STREAM)
        self.set_reuse_addr()
        self.bind(('localhost', port))
        self.address = self.socket.getsockname()
        self.listen(5)