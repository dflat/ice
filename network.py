import socket
import threading
import time
import struct
import queue
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
            return None
        return self.q.get(block=block)


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
        self.size = 512 
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
