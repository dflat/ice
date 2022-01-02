import motion
import time
import console
import threading
import queue
from socket import *
import struct
import math
from data_structures import Record, ButtonState

PORT = 6018
UPDATE_RATE = 1/30
PI = math.pi


def rescale(x, mn=-PI/2, mx=PI/2, a=-128, b=127):
    return a + ((x-mn)*(b-a))/(mx-mn)
    
    
def clamp(x, mn=-128, mx=127):
    return min(mx, max(mn, x))


def find_host():
    q = queue.Queue()
    def find(host, port):
        s = socket(AF_INET, SOCK_DGRAM)
        try:
            s.sendto(b'hi', (host,port))
        except OSError: # assume host is down
            return
        s.setblocking(False)
        attempts = 0
        while attempts < 5:
            try:
                msg, addr = s.recvfrom(12)
                if msg == b'hi':
                    q.put(addr)
                break
            except BlockingIOError:
                time.sleep(.2)
                attempts += 1
     
    for i in range(2, 255):
        host = '192.168.1.%d' % i
        t = threading.Thread(target=find, args=(host,PORT))
        t.start()

    try:
        addr = q.get(timeout=1)
    except queue.Empty:
        addr = None

    return addr
        

    

class ClientConnection:

    def __init__(self, gui):
        self.gui = gui

    # maybe defunct
    def get_button_state(self): # grab from q (size of 1 w/latest state)
        button_state = 0x00
        try:
            button_state = ui_q.get(block=False)
        except queue.Empty:
            pass
        return button_state

    def build_msg(self):
        button_state = self.gui.get_button_state()
        g = motion.get_attitude()
        t = time.time()
        attitude = (int(clamp(rescale(i))) for i in g)
        return struct.pack(Record.fmt, *(attitude), button_state)

    def send_state_to_server(self):
        motion.start_updates()
        
        print('Trying to connect to host...')  # Search for host
        while True:
            host = find_host() # block for 1s max
            if host:
                host = host[0]
                print('Found host.')
                break
            else:
                print('No host found...')
                time.sleep(1)

        s = socket(AF_INET, SOCK_DGRAM)  # Create socket
        
        while True:                      # Send data packets / main net loop
            time.sleep(UPDATE_RATE)
            msg = self.build_msg()
            try:
                s.sendto(msg, (host, PORT))
            except OSError as e:
                print(e)
                break
            except Exception as e:
                s.sendto(b'', (host, PORT))
                print('unhandled exception..')
                raise
            self.gui.clear_button_state()  # TODO: remove this while loop, make a send func
                                    # have a main loop in another file driving both
                                    # this class and Gui class 
        motion.stop_updates()
        print('Capture finished.')
