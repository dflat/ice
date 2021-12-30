import socket
import threading
import time
import struct
import queue

class Record:
    __slots__ = ('roll', 'pitch', 'yaw', 't')
    fmt = 'bbb'

    def __init__(self, roll, pitch, yaw, t=0):
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
        self.t = t
    def __repr__(self):
        return f'Record({self.roll:.2f}, {self.pitch:.2f}, {self.yaw:.2f}'\
                f', t={self.t})'

class NetworkConnection:

    def __init__(self):
        self.remote = threading.Event()
        self.host = ''
        self.port = 6018
        self.backlog = 5
        self.size = 512 
        self.attitude = (0,0,0)
        self.roll, self.pitch, self.yaw = (0,0,0)
        self.last_t = time.time()
        #self.record = Record(0,0,0,self.last_t)
        self.q = queue.Queue()

    def parse_msg(self, msg):
        #print('raw:', msg)
        n_bytes = len(msg) 
        for record_bytes in struct.iter_unpack(Record.fmt, msg):
            yield Record(*record_bytes)

    def get_record(self, block=False):
        #return self.attitude
#        print('qsize:', self.q.qsize())
        if self.q.empty():
            return None
        return self.q.get(block=block)

    def update(self, record):
        #latency = time.time() - record.t
        #print(f'latency: {latency*1000:.0f} ms')
        #print(f'delta time: {delta:.3f}')
        #self.last_t = record.t
        self.q.put(record)
        #print('got:', attitude)
        
    def listen(self):
        t = threading.Thread(target=self._serve)
        t.start()
        print('Listening for data...')

    def shutdown(self):
        self.remote.set()

    def _serve(self):
       # self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.s.bind((self.host,self.port)) 
       # self.s.listen(self.backlog) 
       # client, address = self.s.accept() 
       # print(f'Connected to {address}')
        while not self.remote.is_set(): 
            try:
                data, addr = self.s.recvfrom(self.size) 
                if data: 
                    for record in self.parse_msg(data):
                        #print(record)
                        self.update(record)
                    #client.send(data) 
                else:
                    print('no more data')
                    break
            except KeyboardInterrupt:
                break
            except:
                print('unhandled excpetion!')
                #client.close()
                raise
        #print('closing.')
        #client.close()
