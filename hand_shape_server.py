
from echo_server import EchoServer

import asyncore

class HandShapeServer(object):

    def __init__(self, port):
        self.server = EchoServer(port)

        asyncore.loop()

