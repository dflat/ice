import pygame, pygame.transform
from pygame.surfarray import *
from pygame.locals import *
from numpy import *
from RandomArray import *

RES = array((280, 200))
MAX = 246
RESIDUAL = 86
HSPREAD, VSPREAD = 26, 78
VARMIN, VARMAX = -2, 3

def main():
    "main function called when the script is run"
    #first we just init pygame and create some empty arrays to work with    
    pygame.init()
    screen = pygame.display.set_mode(RES, 0, 8)
    setpalette(screen)
    flame = zeros(RES/2 + (0,3))
    miniflame = pygame.Surface((RES[0]/2, RES[1]/2), 0, 8)
    miniflame.set_palette(screen.get_palette())
    randomflamebase(flame)    

    while 1:
        for e in pygame.event.get():
            if e.type in (QUIT,KEYDOWN,MOUSEBUTTONDOWN):
                return
            
        modifyflamebase(flame)
        processflame(flame)
        blitdouble(screen, flame, miniflame)
        pygame.display.flip()



def setpalette(screen):
    "here we create a numeric array for the colormap"
    gstep, bstep = 75, 150
    cmap = zeros((256, 3))
    cmap[:,0] = minimum(arange(256)*3, 255)
    cmap[gstep:,1] = cmap[:-gstep,0]
    cmap[bstep:,2] = cmap[:-bstep,0]
    screen.set_palette(cmap)


def randomflamebase(flame):
    "just set random values on the bottom row"
    flame[:,-1] = randint(0, MAX, flame.shape[0])


def modifyflamebase(flame):
    "slightly change the bottom row with random values"
    bottom = flame[:,-1]
    mod = randint(VARMIN, VARMAX, bottom.shape[0])
    add(bottom, mod, bottom)
    maximum(bottom, 0, bottom)
    #if values overflow, reset them to 0
    bottom[:] = choose(greater(bottom,MAX), (bottom,0))


def processflame(flame):
    "this function does the real work, tough to follow"
    notbottom = flame[:,:-1]    

    #first we multiply by about 60%
    multiply(notbottom, 146, notbottom)
    right_shift(notbottom, 8, notbottom)

    #work with flipped image so math accumulates.. magic!
    flipped = flame[:,::-1]

    #all integer based blur, pulls image up too
    tmp = flipped * 20
    right_shift(tmp, 8, tmp)
    tmp2 = tmp >> 1
    add(flipped[1:,:], tmp2[:-1,:], flipped[1:,:])
    add(flipped[:-1,:], tmp2[1:,:], flipped[:-1,:])
    add(flipped[1:,1:], tmp[:-1,:-1], flipped[1:,1:])
    add(flipped[:-1,1:], tmp[1:,:-1], flipped[:-1,1:])

    tmp = flipped * 80
    right_shift(tmp, 8, tmp)
    add(flipped[:,1:], tmp[:,:-1]>>1, flipped[:,1:])
    add(flipped[:,2:], tmp[:,:-2], flipped[:,2:])

    #make sure no values got too hot
    minimum(notbottom, MAX, notbottom)


def blitdouble(screen, flame, miniflame):
    "double the size of the data, and blit to screen"
    blit_array(miniflame, flame[:,:-3])
    s2 = pygame.transform.scale(miniflame, screen.get_size())
    screen.blit(s2, (0,0))


if __name__ == '__main__': main()


