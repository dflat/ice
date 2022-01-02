import motion
import time
import console
from socket import *
import struct
import math

HOST = '192.168.1.171'
PORT = 6018
UPDATE_RATE = 1/30
FMT = 'bbb'
PI = math.pi


def rescale(x, mn=-PI/2, mx=PI/2, a=-128, b=127):
	return a + ((x-mn)*(b-a))/(mx-mn)
	
	
def clamp(x, mn=-128, mx=127):
	return min(mx, max(mn, x))


def get_msg():
	g = motion.get_attitude()
	t = time.time()
	vals = (int(clamp(rescale(i))) for i in g)
	return struct.pack(FMT, *(vals))
	
	
def main():
	motion.start_updates()
	print('Trying to connect to host...')
	while True:								# Establish connection
		try:
			s = socket(AF_INET, SOCK_DGRAM)
			#s.connect((HOST, PORT))
			break
		except OSError as e:
			print(e)
			time.sleep(1)
	print('Connected.')
	
	while True:								# Send data packets
		time.sleep(UPDATE_RATE)
		msg = get_msg()
		try:
			s.sendto(msg, (HOST,PORT))
		except OSError as e:
			print(e)
			break
		except Exception as e:
			s.sendto(b'', (HOST,PORT))
			print('unhandled exception..')
			raise
		
		#print(f"roll:{g[0]:.2f}, pitch:{g[1]:.2f}, yaw:{g[2]:.2f}")
		
	motion.stop_updates()
	print('Capture finished.')

if __name__ == '__main__':
	main()
