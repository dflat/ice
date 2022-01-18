import pygame
import itertools
import sys
import random
from pygame.locals import *
import numpy as np

W, H = 400*4, 200*4
SCALE = 1#4
sW, sH = W//SCALE, H//SCALE
G = max(W,H) // SCALE
X = np.linspace(0, 4, G, endpoint=False)
Y = np.linspace(0, 4, G, endpoint=False)
XX,YY = np.meshgrid(X,Y) # shape = (G,G) , e.g. (400,400)
MX = 100
RES = H#600*2
cache = { }

class Snow:
    flake_surfs = []
    colors = [(255,255,255),(200,255,230),(240,220,255)]
    def __init__(self, surf=None, n=200*100):
        self.n = n
#        self.surf = surf
#        self.arr = pygame.surfarray.pixels3d(surf)
        self._init_surfs()
        self.t = 0
        self.x = np.floor(sW*np.random.rand(n))
        self.y = np.floor(sH*np.random.rand(n))
        self.points = np.c_[self.x, self.y]
        #self.masses = np.floor(5 + 5*np.random.rand(n))
        self.gravity = 10
        self.color = (255,255,255)
        self.mx_mass= np.max(self.masses)

    def _init_surfs(self):
        self.flake1 = pygame.Surface((3,3))
        verts = [(1,0),(2,1),(1,2),(0,1)]
        pygame.draw.polygon(self.flake1, self.colors[0], verts)
       # self.flake1 = pygame.transform.scale2x(self.flake1)

        self.flake2 = pygame.Surface((1,1))
        self.flake2.fill(self.colors[1])
        self.flake2b = self.flake2.copy()
        self.flake2b.fill(self.colors[2])

        self.flake3 = pygame.Surface((5,5))
        verts = [(2,0),(4,2),(2,4),(0,2)]
        pygame.draw.polygon(self.flake3, self.colors[2], verts)
       # self.flake3 = pygame.transform.scale2x(self.flake3)

        #self.flake_surfs.extend([self.flake1.convert_alpha()]*3)
        self.flake_surfs.extend([self.flake2.convert_alpha(),
                                 self.flake2b.convert_alpha()]*2)
        #self.flake_surfs.extend([self.flake3.convert_alpha()]*2)
        def area(w,h):return w*h
        self.mass_index = [area(*surf.get_size()) for surf in self.flake_surfs]
        self.mass_index = [m*(1+random.random()) for m in self.mass_index]
        MX_MASS = 5
        self.mass_index = [MX_MASS*m/max(self.mass_index) 
                            for m in self.mass_index]
        k = len(self.flake_surfs)
        self.masses = np.array([self.mass_index[i%k] for i in range(self.n)])
        

    def update2(self, surf, dt):
        self.t += dt
        sf = 4
        self.x += 2*(dt/60/sf)*(self.masses/self.mx_mass) + (.5)*np.sin(self.t/1000)*(self.masses/self.mx_mass)
        self.y += (dt/30/sf)*self.masses #+1 + np.sin(self.t/1000/2)*(self.masses/self.mx_mass)
        #x = np.round(self.x)
        #y = np.round(self.y)
        self.x %= sW
        self.y %= sH
        points = np.c_[self.x,self.y]#np.round(np.c_[x,y].T).astype(np.int32)
        blit_seq = zip(itertools.cycle(self.flake_surfs), points.tolist())
        surf.blits(blit_seq)

    def update(self, surf, i):
        self.x += (1/60)*self.masses + np.sin(i)*(self.masses/self.mx_mass)
        self.y += (1/30)*self.masses + np.sin(i)*(self.masses/self.mx_mass)
        x = np.round(self.x)
        x %= sW
        y = np.round(self.y)
        y %= sH
        points = np.round(np.c_[x,y].T).astype(np.int32)
        arr = pygame.surfarray.pixels3d(surf)
        arr[tuple(points)] = self.color


BLACK = (0,0,0)
WHITE = (255,255,255)
def events():
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

def update(dt):#, arr):
    pass

def draw(draw_screen, render_screen):
    draw_screen.fill(BLACK)
    pygame.draw.circle(draw_screen, WHITE, (10,10), 5)
    scaled_draw_screen = pygame.transform.scale2x(draw_screen)
    render_screen.blit(scaled_draw_screen, (0,0))
    pygame.display.flip()


     
def mod(draw_screen, n):
    arr = pygame.surfarray.pixels3d(draw_screen)
    v = cache.get(n)
    t = 2*np.pi*(n/RES)
    if v is None:
        #z = XX**2 + YY**2
        k = 16*2*np.pi
        z = np.cos(k*(XX - t)) + np.sin(k*(YY + 2 + np.cos(t))) #+ (1/8)*np.cos(2*np.pi*t))
        z *= xx
        z_crop = z[:sW, :sH]
        mx = np.max(z_crop)
        mn = np.min(z_crop)
        #arr[:,:,:] = WHITE
        #arr[:, ::n] = WHITE
        #v = random.randint(1,100)*(z_crop/mx) #+10
        v = 2*(z_crop/mx) #+10
        print('first time n=', n, 'mx=',mx,'mn=',mn)
        cache[n] = v
    arr[:,:,0] = v#[:,::4]
    #arr[:,:,1] = v
    #arr[:,:,2] = v
    #arr[:,:,0] = t*v
    #arr[:,:,1] = v + t
    #arr[:,:,2] = v / t

def run():
    pygame.init()
    render_screen = pygame.display.set_mode((W, H), pygame.DOUBLEBUF, 32)
    draw_screen = pygame.Surface((W//SCALE, H//SCALE))
    #arr = pygame.surfarray.pixels3d(draw_screen)
    fps = 60.0
    fps_clock = pygame.time.Clock()
    dt = 1/fps
    frame = 0
    snow = Snow()
    t = 0
    while True:
        t += dt/1000
        events()
        update(dt)

        draw_screen.fill(BLACK)
        #draw_screen.lock()
        A = MX//2
        SPEED = 8
        n = frame//SPEED % RES  # increment n every SPEED frames, cycling from 0 to RES-1
        #mod2(draw_screen, n)
        snow.update(draw_screen, t)
        render_screen.blit(pygame.transform.scale(draw_screen, (W,H)), (0,0))
        pygame.display.flip()
        print('frame', frame, end='\r')
        frame += 1

        dt = fps_clock.tick(fps)

run()

    
