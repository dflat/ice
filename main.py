import sys
import time
import pygame
import os
from pygame.locals import *
import numpy as np
from collections import deque, defaultdict
import math
import network
import random
import verts

BG_COLOR = (68,52,86)#(213,221,239)#(76,57,79)#(240,240,255)
RED = (255,0,0)
GREY = pygame.Color(100,100,100)#(100,)*3
PLAYER_COLOR = pygame.Color(193,94,152)
WIDTH = 640
HEIGHT = 480
IMAGE_PATH = os.path.join('assets','images')
OBJ_PATH = os.path.join('assets','obj') 

class Explosion(pygame.sprite.Sprite):
    group = pygame.sprite.Group()
    frame_dupes = [3]*10 #[1,1,1,1,1,1,1,1,1,1] #todo...use this
    def __init__(self, pos):
        super().__init__()
        self.group.add(self)
        self.h = 16
        self.w = 16
        self.pos = pos
        self.color = (250,250,250)
        self._load_frames()
        self.frame_no = -1
        self.dupes = 0

    def _load_frames(self):
        self.frames = []
        vframes = verts.parse_obj(os.path.join(OBJ_PATH, 'explosion.obj'))
        self.n_frames = len(vframes)
        print('loaded', self.n_frames)
        for i, vlist in enumerate(vframes):
            alpha = 255*(1 - (i/self.n_frames))
            im = pygame.Surface((self.w,self.h),
                                pygame.SRCALPHA, 32)
            pygame.draw.polygon(im, pygame.Color(*self.color), vlist)
            self.frames.append(im)

    def update(self, dt):
        if self.frame_no < self.n_frames - 1:
            if self.dupes > 0:
                self.dupes -= 1
            else:
                self.frame_no += 1
                self.dupes = self.frame_dupes[self.frame_no]
        else:
            self.kill()

    def draw(self, screen):
        screen.blit(self.frames[self.frame_no], self.pos)

class Drop(pygame.sprite.Sprite):
    group = pygame.sprite.Group()
    COLORS = [pygame.Color(150, 188, 222), pygame.Color(161, 206, 229),
                pygame.Color(169, 217, 231), pygame.Color(187, 228, 233),
                pygame.Color(216, 233, 236), pygame.Color(174, 220, 220)]
    def __init__(self, x, y, height=48, width=16):
        super().__init__()
        self.group.add(self) 
        self.width = width
        self.height = height
        #self.image0 = pygame.Surface([width, height])
        self.image0 = pygame.Surface((width,height),
                                pygame.SRCALPHA, 32)
        self.rect = self.image0.get_rect()

        self._color = random.choice(self.COLORS)
        self.color = self._color
        self.r = self.width
        #self.vertices=[(self.r/3, 2*self.r), (2*self.r- self.r/3, 2*self.r), (self.r, 0)]
        self.vertices = [(0, 0), (width, 0), (width/2, height)]
        pygame.draw.polygon(self.image0, self.color, self.vertices)
        self.image = self.image0.copy()
        #self.image.fill(self.color)

        self.n_ghost_frames = 0
        self._get_ghost_frames()
        #pygame.draw.rect(self.image,
        #                 self.color,
        #                 pygame.Rect(0, 0, width, height))
        self.rect = self.image.get_rect() 
        self.center_offset = np.array([self.width//2,self.height//2], dtype=float)
        self.pos = np.array([x,y], dtype=float)
        self.vel = np.array([0,0], dtype=float)
        self.gravity = random.randint(10,40)
        self.acc = np.array([0,self.gravity], dtype=float)
        self.t = 0
        self.pos_history = deque(maxlen=self.n_ghost_frames)
        self.frame = 0
        self.environ_forces = defaultdict(lambda: np.array([0,0], dtype=float))

    def apply_force(self, force, name):
        self.environ_forces[name] += force

    def clear_force(self, name):
        self.environ_forces[name] = np.array([0,0], dtype=float)

    def rotate(self, phi): # phi is in degrees
        old_center = self.pos
        new_image = pygame.transform.rotate(
                                    self.image0, phi)
        self.image = new_image
        self.rect = self.image.get_rect()
        self.rect.center = old_center
        self.phi = phi

    def update(self, dt):
        self.frame += 1
        if self.pos[1] > HEIGHT + self.height:
            sound.register_miss()
            return self.kill()
        dt /= 1000

        for force in self.environ_forces.values():
            self.acc += force

        #self.vel[1] += self.acc[1]*dt
        #self.pos[1] += self.vel[1]*dt
        self.vel += self.acc*dt
        self.vel[0] *= .90 # horizontal air resistance
        self.pos += self.vel*dt
        self.rect.x = self.pos[0]
        self.rect.y = self.pos[1]
        self.pos_history.append(self.pos.copy())

    def draw(self, screen):
        if self.frame > self.n_ghost_frames:
            for i in range(self.n_ghost_frames):
                screen.blit(self.ghost_images[i], self.pos_history[i])
        screen.blit(self.image, self.pos)

    def _get_ghost_frames(self):
        n = self.n_ghost_frames
        self.ghost_images = []
        shift = .1 if n < 10 else 1/(n)
        self.ghost_colors = [self.color.lerp(BG_COLOR, 1-(i*shift)) for i in range(n)]
        for i in range(n):
            im = self.image.copy()
            pygame.draw.polygon(im, self.ghost_colors[i], self.vertices)
            self.ghost_images.append(im)

class Player(pygame.sprite.Sprite):
    group = set()
    def __init__(self, color=PLAYER_COLOR, height=32, width=64):
        super().__init__()
        self.group.add(self) 
        self.width = width*2
        self.height = height*2
#        self.image = pygame.Surface([width, height])
        self._init_images()
        self.image = self.images[-1]['normal']

        self._color = color
        self.color = color
        #self.image.fill(color)
        #self.image.set_colorkey(COLOR)

        self.n_ghost_frames = 6
        self._get_ghost_frames()
        #pygame.draw.rect(self.image,
        #                 color,
        #                 pygame.Rect(0, 0, width, height))
        self.rect = self.image.get_rect() 
        self.pos = np.array([0,Ice.top - self.height/2], dtype=float)
        self.center_offset = np.array([self.width//2,self.height//2], dtype=float)
        self.t = 0
        self.vel = np.array([0,0], dtype=float)
        self.acc = np.array([0,0], dtype=float)
        self.MAX_VEL = 640*2
        self.MAX_ACC = 1600*2
        self.left = False
        self.right = False
        self.pos_history = deque(maxlen=self.n_ghost_frames)
        self.packet_history = deque(maxlen=4)
        self.missed = 0
        self.x = 0
        self.frame = 0
        self.environ_forces = defaultdict(lambda: np.array([0,0], dtype=float))
        self.direction = -1

    def _init_images(self):
        self.images = defaultdict(dict)
        for phase in ('normal', 'slow'):
            im = pygame.image.load(os.path.join(
                            IMAGE_PATH, f'penguin_{phase}.png')).convert_alpha()
            im = pygame.transform.scale2x(im)
            flipped = pygame.transform.flip(im,True,False)
            self.images[-1][phase] = im 
            self.images[1][phase] = flipped

    def apply_force(self, force, name):
        self.environ_forces[name] += force

    def clear_force(self, name):
        self.environ_forces[name] = np.array([0,0], dtype=float)

    @property
    def center(self):
        return self.pos + self.center_offset

    def _get_ghost_frames(self):
        n = self.n_ghost_frames
        self.ghost_images = defaultdict(lambda: defaultdict(list))
        self.ghost_alphas = [int(40 * ((i+1)/n) ) for i in range(n)]
        for phase in ('normal','slow'):
            for i in range(n):
                im = self.images[-1][phase].copy()
                im.set_alpha(self.ghost_alphas[i])
                self.ghost_images[-1][phase].append(im)
                self.ghost_images[1][phase].append(pygame.transform.flip(im,True,False)) 

    def check_collisions(self):
        hit = pygame.sprite.spritecollideany(self, Drop.group)
        if hit:
            hit.kill()
            sound.play_next()
            Explosion(hit.pos + np.array([0,hit.height]))
             

    def update(self, dt):
        self.dir = 1*self.right - 1*self.left
        self.t += dt
        #self.color = (127+20*math.sin(2*math.pi*self.t/(1000*3)),)*3
        #self.image.fill(self.color)
        record = netcon.get_record()
        #if record is None:
        #    if self.frame > 0:
        #        self.missed += 1
        #    return
        self.frame += 1
        #print('miss rate: %.2f per second' % (self.missed/(self.t/1000)))
        if record:
            self.x = rescale(record.roll, mn=-128, mx=127, a=-1, b=1)
        #print('roll: %.2f, x: %.2f' % (record.roll, x))
#        print(f'roll:{roll:.2f}, pos:{x:.1f}')
        #y = rescale(pitch)
        dt /= 1000
        self.acc[0] = self.x*self.MAX_ACC

        for force in self.environ_forces.values():
            self.acc += force # add environmental forces (e.g. wind)

        # select the correct image to display
        if self.acc[0] > 0:
            self.direction = 1
        else:
            self.direction = -1
        if abs(self.vel[0]) < self.MAX_VEL / 2:
            self.phase = 'slow'
        else:
            self.phase = 'normal'

        self.display_image = self.images[self.direction][self.phase]

        #self.acc[0] += self.dir*self.MAX_ACC/60
        #self.acc[0] = clamp(self.acc[0], -self.MAX_ACC, self.MAX_ACC)
        self.vel[0] += self.acc[0] * dt
        self.vel[0]*= .99 # friction
        self.vel[0] = max(-self.MAX_VEL, min(self.vel[0], self.MAX_VEL))
        #self.pos[0] += 0.5*self.acc[0]*dt*dt + self.vel[0]*dt
        self.pos[0] += self.vel[0]*dt
        print('acc: %.2f, vel: %.2f, pos: %.2f' % (self.acc[0], self.vel[0], self.pos[0]))
        self.pos[1] = Ice.top - self.height/2 # TODO make this dynamic
        self.rect.x = self.pos[0]
        self.rect.y = self.pos[1]
        #self.pos -= self.center_offset

        # apply boundary conditions
        if self.pos[0] > 640:
            self.pos[0] = -self.width #640 - self.center_offset[0]
            #self.vel[0] = 0
        elif self.pos[0] < -self.width:
            self.pos[0] = 640 # - self.center_offset[0]
            #self.vel[0] = 0
        self.pos_history.append(self.pos.copy())
        self.check_collisions()

    def draw(self, screen):
        if self.frame > self.n_ghost_frames:
            for i in range(self.n_ghost_frames):
                screen.blit(self.ghost_images[self.direction][self.phase][i],
                                                self.pos_history[i])
        screen.blit(self.display_image, self.pos)

def vertical_gradient(size, startcolor, endcolor):
    """
    Draws a vertical linear gradient filling the entire surface. Returns a
    surface filled with the gradient (numeric is only 2-3 times faster).
    """
    height = size[1]
    bigSurf = pygame.Surface((1,height)).convert_alpha()
    dd = 1.0/height
    sr, sg, sb, sa = startcolor
    er, eg, eb, ea = endcolor
    rm = (er-sr)*dd
    gm = (eg-sg)*dd
    bm = (eb-sb)*dd
    am = (ea-sa)*dd
    for y in range(height):
        bigSurf.set_at((0,y),
                        (int(sr + rm*y),
                         int(sg + gm*y),
                         int(sb + bm*y),
                         int(sa + am*y))
                      )
    return pygame.transform.scale(bigSurf, size)

class Ice:
    top = 400
    group = set()
    def __init__(self):
        self.group.add(self)
        self.color = pygame.Color(232,238,252,255)
        self.image = vertical_gradient((640,80), BG_COLOR+(255,), self.color)
        #self.image.fill(self.color)
        self.pos = np.array([0,self.top])
        self.rect = self.image.get_rect()
    def draw(self, screen):
        screen.blit(self.image, self.pos) 

class Sound:
    SOUNDS_PATH = os.path.join('assets','sound')
    order = ['ow','shit','fuck','balls','christ']
    def __init__(self):
        pygame.mixer.init()
        pygame.mixer.set_num_channels(32)
        self._load_sounds()
        self.n_sounds = len(self.order)
        self.index = 0
        self.is_combo = False
        self.last_hit_time = 0

    def _load_sounds(self):
        self.sounds = { }
        for path in os.scandir(os.path.join(self.SOUNDS_PATH, 'icicles')):
            name = path.name.split('.')[0]
            sound = pygame.mixer.Sound(path.path)
            self.sounds[name] = sound

    def register_miss(self):
        self.is_combo = False
        self.index = 0

    def play_next(self):
        self.sounds[self.order[self.index]].play()
        self.index = (self.index + 1) % self.n_sounds
        self.is_combo = True
sound = Sound()
         

class Wind:
    def __init__(self):
        self.blowing = False
        self.frame = 0
        self.direction = 1
        self.strength = 1
        self.frequency = 10 # average seconds between gusts of wind
        self.force = np.array([1,0])

    def start(self, n_frames=60*3):
        self.n_frames = n_frames
        self.frame = 0
        self.direction = 1 if random.random() > 0.5 else -1
        self.force = np.array([self.direction*self.strength, 0])
        self.blowing = True

    def update(self):
        if self.blowing:
            print('wind', self.frame)
            self.frame += 1
            if self.frame == self.n_frames:
                self.blowing = False
        else:
            self.chance()

    def chance(self):
        if random.randint(0,60*self.frequency) == 50:
            print('starting wind')
            self.start()
wind = Wind()

def update(dt):
  for event in pygame.event.get():
    if event.type == QUIT:
      netcon.shutdown()
      pygame.quit() # Opposite of pygame.init
      sys.exit() # Not including this line crashes the script on Windows. Possibly
    if event.type == pygame.KEYDOWN:
        if event.key == K_q:
          netcon.shutdown()
          pygame.quit()
          sys.exit()
        elif event.key == K_s:
            #player1.left = True
            pass
        elif event.key == K_f:
            #player1.right = True
            pass
    elif event.type == pygame.KEYUP:
        if event.key == K_s:
            #player1.left = False
            pass
        elif event.key == K_f:
            #player1.right = False
            pass

  if len(Drop.group) < 10:
      Drop(x=random.randint(16, WIDTH-16), y=random.randint(-500, -50))

  wind.update()

  for player in Player.group:
      if wind.blowing:
          player.apply_force(wind.force*2, 'wind')
      else:
          player.clear_force('wind')
      player.update(dt)
 
  for drop in Drop.group:
      if wind.blowing:
          drop.apply_force(wind.force*0.025, 'wind')
          drop.rotate(30*math.sin(math.pi*wind.frame/wind.n_frames))
      else:
          drop.clear_force('wind')
      drop.update(dt)

  for expl in Explosion.group:
      expl.update(dt)

def draw(screen):
  """
  Draw things to the window. Called once per frame.
  """
  screen.fill(BG_COLOR) # Fill the screen with black.

  for ice in Ice.group:
      ice.draw(screen)

  for player in Player.group:
      player.draw(screen)

  for drop in Drop.group:
      drop.draw(screen)

  for expl in Explosion.group:
      expl.draw(screen)

  pygame.display.flip()

netcon = network.NetworkConnection()
netcon.listen()

def rescale(x, mn=-math.pi/2, mx=math.pi/2, a=0, b=WIDTH):
    return a + ((x - mn)*(b-a)) / ( mx - mn)

def clamp(x, mn=-1, mx=1):
    return min(mx, max(x, mn))
 
def run():
  pygame.init()
  fps = 60.0
  fpsClock = pygame.time.Clock()
  
  width, height = WIDTH, HEIGHT
  screen = pygame.display.set_mode((width, height))
  
  player1 = Player()
  ice = Ice()

  dt = 1/fps 
  while True:
    update(dt)
    draw(screen)
    dt = fpsClock.tick(fps)

run()
