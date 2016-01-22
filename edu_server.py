
from echo_server import *
import asyncore
class Edu_Server(object):

    def __init__(self, port):




        self.server = EchoServer(port)

        asyncore.loop()

    #def received_data(self, received_data):
    #    decoded_data = self.converter.decode(received_data)
    #    return self.process_data(decoded_data)




