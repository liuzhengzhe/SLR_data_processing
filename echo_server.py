import asyncore
import asynchat
import logging
import socket
from FrameConverter import FrameConverter
from EduProcessor import EduProcessor
from RecProcessor import RecProcessor
import time
logging.basicConfig(level=logging.DEBUG,format='%(name)s: %(message)s')
import thread
class EchoServer(asyncore.dispatcher):
    def __init__(self, port):
        asyncore.dispatcher.__init__(self)
        self.handlers=[]
        self.timers=[]
        self.create_socket(socket.AF_INET, socket.SOCK_STREAM)
        self.set_reuse_addr()
        self.bind(('', port))
        self.address = self.socket.getsockname()
        #thread.start_new_thread(self.threadfunc,())
        self.listen(5)
        #self.received_data_callback = received_data_callback




    def handle_accept(self):
        client_info = self.accept()
        if client_info is not None:
            sock, addr = client_info
            logging.debug('connecting to %s, tid %s', repr(addr), thread.get_ident())
            self.handler=EchoHandler(client_info[0])
            #self.handlers.append(EchoHandler(client_info[0]))
            #self.timers.append(0)



    def handle_close(self):
        self.close()

    def threadfunc(self):
        start=time.time()
        while 1:
            if time.time()-start==1:
                start=time.time()
                for i in range(len(self.handlers)):
                    self.timers[i]+=1
                    if self.timers[i]>60:
                        try:
                            self.handlers[i].send('heart')
                            self.timers[i]=0
                        except:
                            print 'del',time.time(),i
                            del self.handlers[i]
                            del self.timers[i]
                            print len(self.handlers)
                            break







class EchoHandler(asynchat.async_chat):


    def __init__(self, sock):
        self.state='edu'
        self.received_data = []
        self.eduprocessor = EduProcessor(self.send_data)
        self.recprocessor = RecProcessor(self.send_data)
        # self.received_data_callback = received_data_callback
        self.logger = logging.getLogger('EchoHandler')
        asynchat.async_chat.__init__(self, sock)
        self.set_terminator('#TERMINATOR#')
        self.converter = FrameConverter()


    def collect_incoming_data(self, data):
        #self.logger.debug('collect_incoming_data() -> (%d bytes)\n"""%s"""', len(data), data)
        self.received_data.append(data)

    def found_terminator(self):
        #self.logger.debug('found_terminator()')
        # self.received_data_callback(''.join(self.received_data))
        decoded_data = self.converter.decode(''.join(self.received_data))
        if self.state=='edu':
            self.eduprocessor.process_data(decoded_data)
        elif self.state=='rec':
            self.recprocessor.process_data(decoded_data)
        #if response!=None:
        #self.push(response)
        self.received_data = []

    def send_data(self, data):
        self.send(data)
