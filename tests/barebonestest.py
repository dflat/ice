import pygame, sys
from pygame.locals import *
import time

pygame.init()
#DISPLAYSURF = pygame.display.set_mode((400, 300),)
width, height = 400,300
DISPLAYSURF = pygame.display.set_mode((width, height), pygame.DOUBLEBUF, 32)

pygame.display.set_caption('Hello World!')

loops = 0
loops_this_sec = 0
accum_time = 0
accum_sec = 0
last_time = time.time()
FPS = 300
fps_clock = pygame.time.Clock()
r = pygame.Rect((0,0),(1,1))
while True: # main game loop
    loops += 1
    loops_this_sec += 1
    cur_time = time.time()
    dt = cur_time - last_time
    last_time = cur_time
    accum_time += dt
    accum_sec += dt
    if accum_sec >= 1:
        #print(f'{loops/accum_time:.1f} loops/sec')
        print(f'{loops_this_sec/accum_sec:.1f} loops/sec')
        accum_sec = accum_sec - 1
        loops_this_sec = 0
    #if loops < 1000:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
    #pygame.display.flip()
    #fps_clock.tick(FPS)
    DISPLAYSURF.fill((0,0,0))
    pygame.display.update(r)
