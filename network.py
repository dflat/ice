import socket
import threading
import time
import struct
import queue
import pickle
from collections import defaultdict, deque
from data_structures import Record

class Client:
    def __init__(self, addr):
        self.addr = addr
        self.q = queue.Queue()

    def put(self, item): # used by NetworkConnection instance
        self.q.put(item)

    def get_record(self, block=False): # used by Player instance
        if self.q.empty():
            #print('Empty q from player', (self.addr))
            return None
        n = self.q.qsize()         
        r = self.q.get(block=block)  # grab first record availible
        if n > 1:                    # merge stale records if there is a backlog
            for i in range(n-1):
                r = Record.merge(r, self.q.get())
        return r
        #return self.q.get(block=block)


class NetworkProfiler:
    def __init__(self):
        self.start = time.time()
        self.timings = []
        self.last_time = self.start

    def clock(self, packet):
        t = time.time()
        dt = t - self.last_time
        self.timings.append((dt*1000, packet))
        self.last_time = t
    
    def save_results(self):
        with open('network.prof', 'wb') as f:
            pickle.dump(self.timings, f) 
            
    def load_results(self):
        with open('network.prof', 'rb') as f:
            self.timings = pickle.load(f) 

    def show_results(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        data = [t[0] for t in sorted(self.timings, reverse=True)]
        baseline = [int(1000/30) for i in range(len(data))]
        ax.plot(baseline)
        ax.plot(data)
        data = [t[0] for t in self.timings]
        ax.plot(data)

class NetworkConnection:
    client_id = 0
    clients = { }
    linked_players = { }
    _lock = threading.Lock()
    greeting = b'hi' # todo: abstract this to some config dict

    def __init__(self):
        self.remote = threading.Event()
        self.host = ''
        self.port = 6018
        self.backlog = 5
        self.size = 9#512# 8192
        # Store 10 seconds of packet history per client
        self.record_history = defaultdict(lambda: deque(maxlen=60*10))
        self.last_t = time.time()

    def parse_msg(self, msg):
        n_bytes = len(msg) 
        for record_bytes in struct.iter_unpack(Record.fmt, msg):
            yield Record(*record_bytes)

    def update(self, record, addr):
        #latency = time.time() - record.t
        #print(f'latency: {latency*1000:.0f} ms')
        #print(f'delta time: {delta:.3f}')
        #self.last_t = record.t
        try:
            client = self.clients[addr]
        except KeyError:
            # assume new client
            client = self.register_client(addr) 

        client.put(record)
        self.prof.clock(record)
        self.record_history[addr].append(record)
        
    def listen(self):
        t = threading.Thread(target=self._serve)
        t.start()
        print('Listening for data...')

    def link_player(self, player, remote):
        while not remote.is_set():
            with self._lock:
                for addr in self.clients:
                    if addr in self.linked_players:
                        continue
                    else:        # give player reference to a dedicated client
                        self.linked_players[addr] = id(player)
                        player.establish_link(self.clients[addr])
                        print('linked to player', id(player))
                        return
            print('attempting to link to player', id(player))
            time.sleep(1)

    def register_client(self, addr):
        client = Client(addr)
        self.clients[addr] = client
        return client 

    def shutdown(self):
        self.remote.set()
        #print(self.record_history)

    def greet(self, addr):
        self.s.sendto(b'hi', addr)

    def _serve(self):
        self.prof = NetworkProfiler()
        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.s.bind((self.host,self.port)) 
        while not self.remote.is_set(): 
            try:
                data, addr = self.s.recvfrom(self.size) 
                if data == self.greeting:  # testing this here, in main game loop
                    self.greet(addr)
                elif data == b'':
                    # client disconnected
                    continue
                else:
                    for record in self.parse_msg(data):
                        #print(record)
                        self.update(record, addr)
                    #client.send(data) 
            except KeyboardInterrupt:
                break
            except:
                print('unhandled excpetion!')
                raise
        self.prof.save_results()
