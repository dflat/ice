"""Draw a circle with Numeric"""

#quick and dirty imports
import pygame
from pygame.locals import *
import pygame.surfarray as surfarray
from numpy import *
squareroot = sqrt


def makecircle(radius, color):
    "make a surface with a circle in it, color is RGB"

    #make a simple 8bit surface and colormap
    surf = pygame.Surface((radius*2, radius*2), 0, 8)
    surf.set_palette(((0, 0, 0), color))

    #first build circle mask
    axis = abs(arange(radius*2)-(radius-0.5)).astype(int64)**2
    mask = squareroot(axis[newaxis,:] + axis[:,newaxis])
    mask = less(mask, radius)

    surfarray.blit_array(surf, mask)    #apply circle data
    surf.set_colorkey(0, RLEACCEL)      #make transparent

    return surf


if __name__ == '__main__':
    #lets do a little testing
    from random import *
    pygame.init()
    screen = pygame.display.set_mode((200, 200))
    while not pygame.event.peek([QUIT,KEYDOWN]):
        radius = randint(10, 20)
        pos = randint(0, 160), randint(0, 160)
        color = randint(20, 200), randint(20, 200), randint(20, 200)
        circle = makecircle(radius, color).convert()
        screen.blit(circle, pos)
        pygame.display.update((pos, (radius*2, radius*2)))
        pygame.time.delay(100)
