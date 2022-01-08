import socket
import threading
import time
import datetime
import struct
import queue
import pickle
from collections import defaultdict, deque
#from data_structures import Record

class Client:
    def __init__(self, addr):
        self.addr = addr
        self.q = queue.Queue()

    def put(self, item): # used by NetworkConnection instance
        self.q.put(item)

    def get_record(self, block=False): # used by Player instance
        if self.q.empty():
            return None
        return self.q.get(block=block)

class NetInfo:
    __slots__ = ('packet_no', 'delta_time', 'data_len')
    def __init__(self, *args):
        for i, attr in enumerate(self.__slots__):
            setattr(self, attr, args[i])

class NetworkProfiler:
    host = ''
    port = 8010
    chunk_size = 512
    WINDOW = 1
    POLL_RATE = 1/30
    greeting = b'hi'

    def __init__(self):
        self.packet_no = 0
        self.remote = threading.Event()
        self.timings = []
        self.current_window = []
        self.transfer_rate_over_time = []
        self.packet_rate = 0
        self.time_window = self.WINDOW

    def process(self, data):
        self.packet_no += 1
        dt = self.clock()
        self.time_window -= dt
        # Take a snapshot of the transfer rate every [time_window] seconds.
        if self.time_window < 0: 
            elapsed = 1 - self.time_window # catch remainder
            packets_per_sec = len(self.current_window) / elapsed 
            self.transfer_rate_over_time.append(packets_per_sec)

            fast, normal, lagged = [],[],[]
            for p in self.current_window:
                if p.delta_time < self.POLL_RATE/2 * 1000:
                    fast.append(p)
                elif p.delta_time < self.POLL_RATE*2 * 1000:
                    normal.append(p)
                else:
                    lagged.append(p)

            self.timings.extend(self.current_window) # flush packets and reset window
            self.time_window = self.WINDOW
            self.current_window = []
            self.report_cli(packets_per_sec, fast, normal, lagged)

        self.current_window.append(NetInfo(self.packet_no, dt*1000, len(data)))

    def report_cli(self, packets_per_sec, fast, normal, lagged):
        slowest = 0
        if lagged:
            slowest = list(sorted(i.delta_time for i in lagged))[-1]
        print(f'packets / second: {packets_per_sec:<5.1f}'
              f'[fast: {len(fast):<3}normal: {len(normal):<3}'
              f'lagged: {len(lagged):<3}slowest: {slowest:<5.1f}ms]')
            
    def serve_forever(self):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.s.bind((self.host, self.port)) 
        # try nonblocking/select TODO
        print('Waiting for client to start sending data...')
        self.s.settimeout(5)
        while not self.remote.is_set(): 
            try:
                data, addr = self.s.recvfrom(self.chunk_size) 
                if data == self.greeting:
                    self.greet(addr)
                    self.start_clock()
                    print(f'Started monitoring transfer rate with {addr[0]}')
                elif data == b'':
                    print('client disconnected')
                    continue
                elif data == b'bye':
                    print('client disconnected w/ bye message.')
                    break
                else:
                    self.process(data)
                    #for record in self.parse_msg(data):
                        #print(record)
                        #self.update(record, addr)
                    #client.send(data) 
            except KeyboardInterrupt:
                break
            except OSError:
                print('probably timed out')
                break
            except:
                print('unhandled excpetion!')
                raise
        self.save_results()

#    def send_forever(self):

    def start_clock(self):
        self.start = time.time()
        self.last_time = self.start

    def clock(self):
        t = time.time()
        dt = t - self.last_time
        self.last_time = t
        return dt
    
    def save_results(self):
        with open('network.prof', 'wb') as f:
            data = (self.timings, self.transfer_rate_over_time)
            pickle.dump(data, f) 
            
    def load_results(self):
        with open('network.prof', 'rb') as f:
            self.timings, self.transfer_rate_over_time = pickle.load(f) 

    def show_results(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        data = [t[0] for t in sorted(self.timings, reverse=True)]
        baseline = [int(1000/30) for i in range(len(data))]
        ax.plot(baseline)
        ax.plot(data)
        data = [t[0] for t in self.timings]
        ax.plot(data)

    def greet(self, addr):
        self.s.sendto(b'hi', addr)

    def shutdown(self):
        self.remote.set()

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
        # todo: try nonblocking socket (and use select module polling)
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

if __name__ == '__main__':
    profiler = NetworkProfiler()
    profiler.serve_forever()

