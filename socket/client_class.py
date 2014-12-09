import asyncore
import asynchat
import logging
import socket

class EchoClient(asynchat.async_chat):

    def __init__(self, host, port):
        self.received_data = []
        #self.handler = handler
        self.logger = logging.getLogger('EchoClient')
        asynchat.async_chat.__init__(self)
        self.create_socket(socket.AF_INET, socket.SOCK_STREAM)
        self.logger.debug('connecting to %s', (host, port))
        self.connect((host, port))
        #self.set_terminator('\n')