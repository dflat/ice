import sys
import time
import pygame
import itertools
import threading
import os
from pygame.locals import *
import numpy as np
from collections import deque, defaultdict
import math
import network
import random
from heapq import heappush, heappop
from PIL import Image, ImageFilter
import verts
from utils.audio_tools import WaveStream

BG_COLOR = (68,52,86)#(213,221,239)#(76,57,79)#(240,240,255)
RED = (255,0,0)
PURPLE = (200,0,200)
GREY = pygame.Color(100,100,100)#(100,)*3
WHITE = (255,255,255)
PLAYER_COLOR = pygame.Color(193,94,152)
WIDTH = 1920#1200
HEIGHT = 800#480
SECTOR_WIDTH = 300 # divide up screen into virtual columns (sectors)
IMAGE_PATH = os.path.join('assets','images')
OBJ_PATH = os.path.join('assets','obj') 

def load_obj_frames(w, h, color, obj_filename, double_size=0):
    frames = []
    vframes = verts.parse_obj(os.path.join(OBJ_PATH, obj_filename))
    n_frames = len(vframes)
    print('loaded', n_frames)
    for i, vlist in enumerate(vframes):
        if isinstance(color, list):
            c = color[i]
        else:
            c = color
        alpha = 255*(1 - (i/n_frames))
        im = pygame.Surface((w,h),
                            pygame.SRCALPHA, 32)
        pygame.draw.polygon(im, pygame.Color(*c), vlist)
        for _ in range(double_size):
            im = pygame.transform.scale2x(im) # todo: testing this
        if double_size == -1:
            im = pygame.transform.scale(im, (w//2,h//2))
        frames.append(im)
    return frames

def swap_palette(im, old_c, new_c):
    surf = pygame.Surface(im.get_size())
    surf.fill(new_c)
    im.set_colorkey(old_c)
    surf.blit(im, (0,0))
    return surf
def palette_swap(im, old_c, new_c):
    x,y = im.get_size()
    for i in range(x):
        for j in range(y):
            col = im.get_at((i,j))
            if col == old_c:
                im.set_at((i,j), new_c)


class Explosion(pygame.sprite.Sprite):
    group = pygame.sprite.Group()
    frame_dupes = [3]*10
    w = 16
    h = 16
    color = (250,250,250)
    frames = load_obj_frames(w, h, color, 'explosion.obj', double_size=1)
    n_frames = len(frames)

    def __init__(self, pos):
        super().__init__()
        self.group.add(self)
        self.pos = pos
        self.frame_no = -1
        self.dupes = 0
        self.boundary = self.n_frames - 1

    def update(self, dt):
        if self.frame_no < self.boundary:
            if self.dupes > 0:
                self.dupes -= 1
            else:
                self.frame_no += 1
                self.dupes = self.frame_dupes[self.frame_no]
        else:
            self.animation_finished()

    def animation_finished(self):
        self.kill()

    def draw(self, screen):
        screen.blit(self.frames[self.frame_no], self.pos)

class ProjectileExplosion(Explosion):
    w = 16
    h = 16
    color = (200,70,100)
    frames = load_obj_frames(w, h, color, 'explosion.obj', double_size=1)

class SnowPlume(Explosion):
    w = 32
    h = 8
    color = [pygame.Color(232,238,252).lerp(BG_COLOR, .1 - .02*i) for i in range(5)]
    frames = load_obj_frames(w, h, color, 'snow_plume.obj', double_size=2)
    flipped = [pygame.transform.flip(im,True,False) for im in frames]
    n_frames = len(frames)
    frame_dupes = [2]*n_frames
    offset_shift = np.array([12, 32])
    flipped_offset_shift = np.array([-offset_shift[0], offset_shift[1]])
    active = { }
    frame_seq = list(range(n_frames)) + list(reversed(range(n_frames)))[1:-1]

    def __init__(self, pos, player):
        super().__init__(pos) #  TESTING
        self.offset = self.offset_shift - player.center_offset
        self.flipped_offset = self.flipped_offset_shift - player.center_offset
        self.player = player
        self.active[id(player)] = self
        self.killed = False
        self.frame_gen = self.get_next_frame()

    def update(self, dt):
        prev_frame_no = self.frame_no
        self.frame_no = next(self.frame_gen)
        if self.killed and self.frame_no == 1 and prev_frame_no == 0:
            self.kill()

    @classmethod
    def deactivate(cls, player):
       cls.active[id(player)].finish() 

    def get_next_frame(self):
        for frame_no in itertools.cycle(self.frame_seq):
            for active_frame in itertools.repeat(frame_no, self.frame_dupes[frame_no]):
                yield active_frame
        
    def animation_finished(self):
        self.frame_no = 0 # this will loop forever, todo: ping pong

    def finish(self):
        self.killed = True

    def ping_pong(self, start_frame, end_frame):
        pass

    def draw(self, screen):
        if self.player.direction == 1:
            frames = self.frames
            offset = self.offset
        else:
            frames = self.flipped
            offset = self.flipped_offset 
        screen.blit(frames[self.frame_no], self.pos + offset)

class Twinkle(Explosion):
    w = 16
    h = 16
    color = (250,250,250)
    frames = load_obj_frames(w, h, color, 'explosion.obj', double_size=-1)
    frame_dupes = [1]*10

    def __init__(self, drop: 'Drop'):
        super().__init__(drop.pos)
        self.drop = drop
        #self.offset = np.array([6*random.random(), 24*random.random()])
        self.offset = np.array([random.randint(0,12) - 6, random.randint(0,12) - 6])

    def draw(self, screen):
        screen.blit(self.frames[self.frame_no], self.pos + self.offset)

class Drop(pygame.sprite.Sprite):
    group = pygame.sprite.Group()
    COLORS = [pygame.Color(150, 188, 222), pygame.Color(161, 206, 229),
                pygame.Color(169, 217, 231), pygame.Color(187, 228, 233),
                pygame.Color(216, 233, 236), pygame.Color(174, 220, 220)]

    _image_cache = { }
    _sectors = [] # heap tracking which column of screen Drops are populated

    def __init__(self, x=0, y=0, height=48, width=16):
        super().__init__()
        self.group.add(self) 
        self.assign_sector()
        self.width = width
        self.height = height
        self.color_id = random.randint(0, len(self.COLORS)-1)
        self._color = self.COLORS[self.color_id]
        self.color = self._color
        self.image0 = self._load_image()
        self.image = self.image0.copy()
        self.rect = self.image.get_rect()

        self.n_ghost_frames = 0 #TODO bug with ghost frames
        self._get_ghost_frames()
        self.center_offset = np.array([self.width//2,self.height//2], dtype=np.int64)
        self.pos = np.array([self.x,y], dtype=float)
        self.vel = np.array([0,0], dtype=float)
        self.gravity = random.randint(10,40)#(10,40)
        self.acc = np.array([0,0], dtype=float)
        self.falling = True
        self.t = 0
        self.pos_history = deque(maxlen=self.n_ghost_frames)
        self.frame = 0
        self.environ_forces = defaultdict(lambda: np.array([0,0], dtype=float))
        self.register_force(np.array([0, self.gravity]), 'gravity')  
        self.register_force(game.wind.force, 'wind') # register wind force
        self.twinkle_freq = random.randint(60*1, 60*2)
        self.max_rotation = random.randint(10, 40)
        self.type = 'regular'
        self.mana = 1

    @classmethod
    def _init_sectors(cls, n=WIDTH//SECTOR_WIDTH):
        """
        keep a heap of 3-item lists,
        tracking how many drops populate each sector (column) of
        the screen. use a random int as the second entry for the 
        dual purpose of random tie-breakers for equally populated sectors,
        and a random intra-sector pixel offset amount.
        looks like:
            [# of drops in sector, randint(pad, SECTOR_WIDTH-pad), sector index].
        """
        for i in range(n):
            heappush(cls._sectors, [0,random.randint(30,SECTOR_WIDTH-30),i])

    def assign_sector(self):
        s = heappop(self._sectors) 
        s[0] += 1 
        s[1] = random.randint(30, SECTOR_WIDTH - 30)
        self.x = s[2]*SECTOR_WIDTH + s[1]
        heappush(self._sectors, s)
        print(self._sectors)
        
    def _load_image(self):
        im = self._image_cache.get((self.width,self.height,tuple(self.color)))
        if not im: 
            w, h = self.width/2, self.height/2
            im = pygame.Surface((w,h), pygame.SRCALPHA, 32).convert_alpha()
            self.vertices = [(0, h/7), ((w-1)/3, 1), ((2/3)*(w-1),1),
                                (w-1, h/8), ((w-1)/2, h-1)]
            pygame.draw.polygon(im, self.color, self.vertices)
            shading_verts = [((2/3)*(w-1),1),
                                (w-1, h/8), ((w-1)/2, h-1)]
            pygame.draw.polygon(im, self.color.lerp((255,255,255), .25), shading_verts)
            im = pygame.transform.scale2x(im)
            self._image_cache[(w,h,tuple(self.color))] = im
        return im
        #self.image0 = im
        #self.image = im.copy()

    def register_force(self, force, name):
        self.environ_forces[name] = force

    def clear_force(self, name):
        #self.environ_forces[name] = np.array([0,0], dtype=float)
        self.environ_forces.pop(name)

    def rotate(self, phi): # phi is in degrees
        old_center = self.pos
        #new_image = self.rotate_image(self.image0, phi)
        new_image = pygame.transform.rotate(self.image0, phi)
        self.image = new_image
        self.rect = self.image.get_rect()
        self.rect.center = old_center
        self.phi = phi

    def rotate_image(self, im, phi):
        im = self._image_cache.get((self.color_id, phi))
        if not im:
            im = pygame.transform.rotate(self.image0, phi)
            self._image_cache[(self.color_id, phi)] = im
        else:
            print(f'cached ({len(self._image_cache)}): phi:{phi}, color:{self.color_id}')
        return im

    def sum_of_forces(self):
        forces = np.array([0,0], dtype=float)
        for force in self.environ_forces.values():
            forces += force
        return forces

    def update_pos(self): 
        self.rect.center = self.pos

    def update(self, dt): # Drop
        self.frame += 1

        # check if off-screen
        if self.pos[1] > HEIGHT + self.height:
            return self.kill()

        # check for wind force
        if game.wind.blowing:
            #self.rotate((self.max_rotation/game.wind.strength)*game.wind.force[0])
            # rotate to match velocity tangent 
            pass
        self.rotate(90 - np.angle(complex(*normalize((self.vel))), deg=True))


        self.acc = self.sum_of_forces() #/ self.mass

        # integrate physics
        self.pos += self.vel*dt # testing pos integr. before vel
        self.vel += self.acc*dt
        self.vel[0] *= .90 # horizontal air resistance

        # check if hit ground
        if self.falling and self.rect.bottom > game.ice.mid:
            self.falling = False
            self.pos[1] -= self.vel[1]*dt # roll back position
            self.vel[1] = 0
            self.register_force(np.array([0, -self.gravity]), 'ground normal')
            print('icicle landed')
        
        #self.pos += game.cam.pos # TESTING
        self.update_pos()


        # remember position history
        self.pos_history.append(self.pos.copy())

        # special animations
        if self.frame % self.twinkle_freq == 0:
            Twinkle(self)# + np.array([0, self.height*random.random()]))

    def draw(self, screen):
        if self.frame > self.n_ghost_frames:
            for i in range(self.n_ghost_frames):
                screen.blit(self.ghost_images[i], self.pos_history[i])
        screen.blit(self.image, self.rect.topleft + game.cam.offset)
        if game.debug_rects:
            pygame.draw.rect(screen, WHITE, self.rect, width=1)
            tangent = 20*normalize((self.vel))
            pygame.draw.line(screen, PURPLE, self.pos, self.pos + tangent) 

    def _get_ghost_frames(self):
        n = self.n_ghost_frames
        self.ghost_images = []
        shift = .1 if n < 10 else 1/(n)
        self.ghost_colors = [self.color.lerp(BG_COLOR, 1-(i*shift)) for i in range(n)]
        for i in range(n):
            im = self.image.copy()
            pygame.draw.polygon(im, self.ghost_colors[i], self.vertices)
            self.ghost_images.append(im)


class Arrow(Drop):
    group = pygame.sprite.Group()
    power = 5

    def __init__(self, player):
        super().__init__()
        self.group.add(self) 
        self.player = player
        self.theta = 0
        self.inner_r = 80

        self.ring_rect = pygame.Rect(player.rect.center, (self.inner_r*2, self.inner_r*2))
        self.arch_rect = pygame.Rect(player.rect.center,
                                    (self.inner_r*2 + 10, self.inner_r*2))
        
        # Test initializing these attributes here to allow projectile
        # to fire on first frame of existence, before a single update takes place
        self.aim_vector = np.array([self.player.direction*math.cos(self.theta),
                                            math.sin(self.theta)])
        self.tip_offset = self.inner_r*self.aim_vector
        self.aim_center = self.ring_rect.center + np.array(self.tip_offset)

        self.width = 16 # arrow width
        self.height = 48 # arrow height
        self._color = random.choice(self.COLORS)
        self.color = self._color
        self.image0 = self._load_image()
        self.image = self.image0.copy()
        self.rect = self.image.get_rect()

        self.rect.center = self.aim_center#.copy() # TEST

        self.offset = np.array((-8,-24))#-self.rect.x, -self.rect.y))
        self.pos_history = deque(maxlen=60)
        self.gravity = 20*60
        self.speed = 30*60
        self.pos = np.array(self.rect.topleft, dtype='float') # .center testing...
        self.frame = 0
        self.fired = False
        self.time_since_fired = 0

    def update_pos(self):
        self.rect.center = self.pos #[0]
        #self.rect.y = self.pos[1]

    def update(self, dt): # Arrow
        #dt /= 1000
        if self.fired:
            self.time_since_fired += dt

            # integrate projectile motion
            self.vel[1] += self.gravity*dt
            self.rect.x += self.vel[0]*dt
            self.rect.y += self.vel[1]*dt
            self.pos += self.vel*dt
            self.update_pos()

            # track projectile center for tail trace
            pos = self.rect.center 
            self.pos_history.append(pos)
            
            # check boundary conditions
            if pos[1] > HEIGHT + 900 or pos[0] > WIDTH + 900 or pos[0] < -900: #300
                self.kill()

            self.collision_with_players_check()
            self.collision_with_other_arrows_check()

            # rotate to velocity tangent
            self.rotate(90 + np.angle(complex(*normalize(flip_y(self.vel))), deg=True))

        else:
            self.frame += 1
            r = self.inner_r
            self.ring_rect.center = self.player.rect.center
            self.arch_rect.center = self.player.rect.center
            self.theta = rescale(self.player.dy,mn=-128,mx=127,a=-math.pi/2,b=math.pi/2)
            self.aim_vector = np.array([self.player.direction*math.cos(self.theta),
                                            math.sin(self.theta)])
            self.tip_offset = r*self.aim_vector
            self.aim_center = self.ring_rect.center + np.array(self.tip_offset)
            self.rect.center = self.aim_center #.copy()
            self.pos = self.aim_center.copy()
            self.rotate(90*self.player.direction
                        - self.player.direction*self.theta*(180/math.pi))
        
    def collision_with_players_check(self):
        for player in Player.group - {self.player}:
            if player.rect.collidepoint(self.rect.center):
                #player.take_damage()

                #  Transfer velocity to hit player. If the shot is fired
                #  from above, give the hit player a recoil kick up. 
                arrow_v, power = self.vel, self.power
                player.vel[0] += arrow_v[0]*power
                y_vel = -arrow_v[1] if arrow_v[1] > 0 else arrow_v[1]
                player.vel[1] += y_vel*power 

                ## Remove arrow sprite, animate hit
                self.kill()
                ProjectileExplosion(self.rect.center)
                game.sound_fx.play_congrats()
                game.cam.start_shake()
    
    def handle_collision_with_arrow(self, arrow, intersection_point):
        print('arrows intersect')
        game.sound_fx.start_fx('intercepted')
        arrow.kill()
        self.kill()
        ProjectileExplosion(self.pos+intersection_point)

        # If player uses unfired arrow as a 'shield'
        # to intercept an arrow fired by another player.
        if Player.active_slowmo is arrow.player: 
            game.exit_slow_mo(arrow.player)

        # Award points to whoever fired second,
        # as they got the 'interception'; If the player
        # is loaded but hasn't fired, that counts as
        # 'firing second'.
        if self.time_since_fired < arrow.time_since_fired:
            self.player.intercepted_arrow()
        else:
            arrow.player.intercepted_arrow()

    def collision_with_other_arrows_check(self):
        player_who_fired = self.player
        enemy_arrows = {a for a in Arrow.group if a.player != player_who_fired}
        for arrow in enemy_arrows:
            # Do crude bounding box check
            if arrow.rect.colliderect(self.rect):
                ## todo: or skip masks and just do a circle intersection
                print('arrows AABB intersect')
                # Do pixel perfect check, use a mask
                mask = pygame.mask.from_surface(self.image)
                other = pygame.mask.from_surface(arrow.image)
                offset = (arrow.rect.x - self.rect.x, arrow.rect.y - self.rect.y)
                intersection = mask.overlap(other, offset)
                if intersection:
                    self.handle_collision_with_arrow(arrow, intersection)

    def fire(self):
        self.fired = True
        self.vel = self.speed*self.aim_vector

    def rotate(self, phi): # phi is in degrees
        old_center = self.rect.center
        new_image = pygame.transform.rotate(
                                    self.image0, phi)
        self.image = new_image
        self.rect = self.image.get_rect()
        self.rect.center = old_center
        self.phi = phi

    def draw(self, screen):
        d = self.player.direction

        ## Draw overlay
        if not self.fired:
            pygame.draw.arc(screen, self.color, self.ring_rect, d*-math.pi/2, d*math.pi/2,
                        width = random.randint(1,3))
            pygame.draw.arc(screen, self.color, self.arch_rect, d*-math.pi/2, d*math.pi/2,
                        width = random.randint(1,3))
            pygame.draw.circle(screen, (255,255,255),
                                self.player.rect.center + self.tip_offset, 4)

        ## Draw projectile
        screen.blit(self.image, self.rect.topleft) 
        pygame.draw.circle(screen, (255,255,255), self.aim_center, 4)
        self.draw_trace(screen)

        if game.debug_rects:
            pygame.draw.rect(screen, WHITE, self.rect, width=1)
            tangent = 20*normalize((self.vel))
            pygame.draw.line(screen, PURPLE, self.pos, self.pos + tangent) 

    def draw_trace(self, screen):
        #offset = self.aim_center + self.offset
        n = len(self.pos_history)-1
        for i in range(n):
            if (0 <= self.pos_history[i][0] < WIDTH - 0
            and 0 <= self.pos_history[i][1] < HEIGHT):
                #bg_color = game.bg_image.get_at(self.pos_history[i]) # BG_COLOR
                bg_color = BG_COLOR #if self.pos_history[i][1] < HEIGHT - 80 else WHITE
                pygame.draw.line(screen, pygame.Color(255,255,255).lerp(bg_color,(1-i/n)),
                                 self.pos_history[i], self.pos_history[i+1])

class Shield(Arrow):
    def handle_collision_with_arrow(self, arrow, intersection_point):
        pass

def assemble_image(surf, obj_filename, color_map):
    path = os.path.join(OBJ_PATH, obj_filename)
    parts = verts.parse_obj_as_dict(path)
    for key in color_map: 
        pygame.draw.polygon(surf, color_map[key], parts[key][0])
    return surf

class PlayerBubble(pygame.sprite.Sprite):
    group = pygame.sprite.Group()
    w = 72
    h = 72
    resolutions = 20
    #_colors = {1: (255,0,0), 2: (0,0,255)}
    #player_images = { }

    def __init__(self, player):
        super().__init__()
        self.group.add(self)
        self.player = player
        self.pos = np.array((player.rect.x, 0))
        #self.image = self.player_images[player.player_id]
        self._gen_image()
        #self.rect = self.image.get_rect()
        self.visible = False

    #def update_pos(self):
        #self.rect.topleft = self.pos

    def _gen_image(self):
        pad = 10
        w, h = self.w + 2*pad, self.h + 2*pad
        center = (w/2, h/2)
        color = (220,220,220)#pygame.Color(BG_COLOR).lerp(Player.colors[self.player.player_id], .8)
        inner_color = BG_COLOR
        tip_points = [(w/2,0), (w/2 - w/8, h/4), (w/2 + w/8, h/4)]
        aspect = self.player.width / self.player.height
        thumbnail = pygame.transform.scale(self.player.image,
                    (self.player.width//2, self.player.height//2))
        thumb_w, thumb_h = thumbnail.get_size()
        image = pygame.Surface((w, h)).convert_alpha()
        pygame.draw.circle(image, color, center, self.w/2) # outer circle
        pygame.draw.polygon(image, color, tip_points)      # triangle tip
        pygame.draw.circle(image, inner_color, center, self.w/2 - 4) # inner circle
        image.set_alpha(200)
        self.images = []
        for i in range(self.resolutions):
            res_h = thumb_h - i
            res_w = res_h*aspect 
            im = image.copy()
            thumb_im = pygame.transform.scale(thumbnail, (int(res_w), int(res_h)))
            tw, th = thumb_im.get_size()
            im.blit(thumb_im, (center[0] - tw/2, center[1] - th/2))
            self.images.append(im)

        #self.player_images[player_id] = image
        #self.image0 = image
        #self.image = image.copy()
        
    def update(self, dt):
        if self.player.rect.bottom > 0:
            self.visible = False
        else:
            self.visible = True
            self.pos[0] = self.player.rect.x
            dist_above_top = -self.player.rect.bottom
            mx_index = self.resolutions - 1
            index = int(rescale(dist_above_top, mn=1, mx=200, a=0, b=mx_index))
            self.image = self.images[9]#clamp(index, mn=0, mx=mx_index)]
            #print('dist above:', dist_above_top, 'i:', index)

    def draw(self, screen):
        if self.visible:
            screen.blit(self.image, self.pos)


class Player(pygame.sprite.Sprite):
    players = 0
    dropped_packets = 0
    sound_packs = {1: ['penguin', 'guitar'], 2:['penguin', 'bass']}
    SOUND_RESET_TIMEOUT = 2 # seconds to restart sound order
    collisions = { }
    id_map = { }
    collision_pairs = { }
    active_slowmo = None
    group = set()
    n_phases = 12
    phases = ['1','2','3','4'] + ['5']*(n_phases - 3)
    phase_map = dict(zip(range(len(phases)), phases))
    colors = {1: pygame.Color(39,36,41),
              2: pygame.Color(200,0,255).lerp((39,36,41), .8),
    }
    W = 64*2
    H = 32*2

    def __init__(self, color=PLAYER_COLOR, width=64, height=32, pos_x=0, skin=None):
        super().__init__()
        self.group.add(self) 
        self.width = width*2
        self.height = height*2
        self.skin = skin
        self._init_images()
        self._init_reflections()
        self.image = self.images[-1]['1']
        self.rect = self.image.get_rect() 

        self._color = color
        self.color = color

        self.n_ghost_frames = 2
        self._init_ghost_frames()
        self.pos_history = deque(maxlen=self.n_ghost_frames+10)
        self.packet_history = deque(maxlen=4)

        self.t = 0
        self.pos = np.array([pos_x, Ice.top], dtype=float)
        self.center_offset = np.array([self.width//2,self.height//2], dtype=np.int64)
        self.vel = np.array([0,0], dtype=float)
        self.acc = np.array([0,0], dtype=float)
        self.acc_x_input = 0
        self.MAX_VEL = 640*2
        self.MAX_ACC = 1600*2

        self.missed = 0
        self.frame = 0
        self.friction = 0.97 # 0.99 Test out best value ..
        self.environ_forces = defaultdict(lambda: np.array([0,0], dtype=float))
        self.register_force(game.wind.force, 'wind') # register wind force
        self.direction = -1
        self.phase = self.phases[0]
        self.getting_hit = False
        self.recoil_cooldown_frames = 0
        self.jumping = False
        self.slow_mo_dur = 0
        self.last_hit = 0
        self.last_seq_no = 0
        self.slowmo_enters = 0
        self.slowmo_exits = 0
        self.slowmo_triggers = 0
        self.mana = 0
        self.ammo = defaultdict(int)

        self._register_new_player()
        self._connect_to_network()

    def _connect_to_network(self):
        self.client = None 
        self.remote = threading.Event()
        t = threading.Thread(target=netcon.link_player, args=(self,self.remote))
        t.start()

    def establish_link(self, client): # will be called by a the server
        self.client = client

    def disestablish_link(self):      # will be called by the server
        """
        Server will initiate after receiving disconnect message from client;
        Then, attempt to re-establish connection to player in a new thread.
        """
        print(f'Player {id(self)} disconnected')
        self._connect_to_network()

    @classmethod
    def _update_collision_graph(cls):
        for pair in itertools.combinations(range(1, cls.players + 1), 2):
            a, b = pair
            cls.collision_pairs[Player.id_map[a]] = Player.id_map[b]

    def _register_new_player(self):
        Player.players += 1
        self.player_id = Player.players
        Player.id_map[self.player_id] = self
        Player._update_collision_graph()
        PlayerBubble(self)
        self.sound = Sound(asset_packs = self.sound_packs[self.player_id],
                            instrument = self.player_id)

    def _init_images(self):
        self.images = defaultdict(dict)
        for phase in set(self.phases):
            im = pygame.image.load(os.path.join(
                            IMAGE_PATH, f'penguin_{phase}.png')).convert_alpha()
            if self.skin:
                skin_im = pygame.image.load(os.path.join(
                                IMAGE_PATH, f'{self.skin}.png')).convert_alpha()
                im.blit(skin_im, (0,0))
                palette_swap(im, (39,36,41), pygame.Color(200,0,255).lerp((39,36,41), .8))
            # palette swap
            palette_swap(im, (0,0,0), pygame.Color(0,0,0).lerp(BG_COLOR, .5))
            im = pygame.transform.scale2x(im)
            flipped_x = pygame.transform.flip(im, True, False)
            self.images[-1][phase] = im 
            self.images[1][phase] = flipped_x
        self.hit_im = pygame.Surface(im.get_size())
        self.hit_im.fill((255,0,0))

    def get_reflection_image(self, intensity):
        return self.images[self.direction << intensity+1][self.phase]

    def _init_reflections(self):
        self.max_reflection_resolution = 10
        n = self.max_reflection_resolution
        alphas = [int(60 * (i/n) ) for i in range(n)] # 0 is transparent
        for phase in set(self.phases):
            im = self.images[-1][phase].copy() 
            flipped_x = self.images[1][phase].copy() 
            flipped_y =  pygame.transform.flip(im, False, True)
            flipped_x_and_y = pygame.transform.flip(flipped_x, False, True)
            for i in range(self.max_reflection_resolution):
                im_fy = flipped_y.copy()

                # apply progressive blur and alpha to each reflected image
                im_bytes = pygame.image.tostring(im_fy, 'RGBA', False)
                pil_im = Image.frombytes('RGBA', im_fy.get_size(), im_bytes)
                pil_im = pil_im.filter(ImageFilter.BoxBlur(radius=13-i))
                im_fy = pygame.image.fromstring(pil_im.tobytes(),pil_im.size,'RGBA')

                im_fxy = pygame.transform.flip(im_fy, True, False)
                im_fy.set_alpha(alphas[i])
                im_fxy.set_alpha(alphas[i])

                # lazy bit-shift indexing hack to avoid making a new dict layer
                self.images[-1 << i+1][phase] = im_fy
                self.images[1 << i+1][phase] = im_fxy

    def _init_ghost_frames(self):
        n = self.n_ghost_frames
        self.ghost_images = defaultdict(lambda: defaultdict(list))
        self.ghost_alphas = [int(40 * ((i+1)/n) ) for i in range(n)]
        for phase in set(self.phases):
            for i in range(n):
                im = self.images[-1][phase].copy()
                im.set_alpha(self.ghost_alphas[i])
                self.ghost_images[-1][phase].append(im)
                self.ghost_images[1][phase].append(pygame.transform.flip(im,True,False)) 

    def register_force(self, force, name):
        self.environ_forces[name] = force

    def clear_force(self, name):
        #self.environ_forces[name] = np.array([0,0], dtype=float)
        self.environ_forces.pop(name)

    @property
    def topleft(self):
        return self.pos - self.center_offset

    def check_drop_collisions(self):
        hit = pygame.sprite.spritecollideany(self, Drop.group)
        if hit:
            t = time.time()
            if t - self.last_hit > self.SOUND_RESET_TIMEOUT:
                self.sound.register_miss()
            hit.kill()
            self.sound.play_next()
            self.last_hit = t
            Explosion(hit.pos + np.array([0,hit.height]))
            self.update_inventory(hit)

    # TODO: fix this hack
    def check_collision_with_ground(self):
        if self.vel[1] > 0 and self.pos[1] >= Ice.top:# + 2:
            self.jumping = False
            #print('stopped')
            self.vel[1] = 0
        self.pos[1] = min(self.pos[1], Ice.top) # dont fall thru floor

    def intercepted_arrow(self):
        print(f'Player {self.player_id} intercepted an arrow.')
        self.mana += 3

    def update_inventory(self, drop):
        self.mana += drop.mana
        self.ammo[drop.type] += 1
        
    def update_pos(self):
        self.rect.center = self.pos

    def check_next_pos(self):
        pass

    @classmethod
    def check_for_collisions_between_players(cls, recoil_frames=6):
        '''
        Checks for collisions between unique pairs of players
        '''
        for a, b in cls.collision_pairs.items():
            a_fut_pos = a.pos + a.vel*(1/60000)
            b_fut_pos = b.pos + b.vel*(1/60000)

            # update rects and collide test, one component at a time

            # test x collision
            a.rect.x = a_fut_pos[0]
            b.rect.x = b_fut_pos[0]
            would_hit_x = pygame.sprite.collide_rect(a,b) 

            # restore x component
            a.rect.x = a.pos[0]
            b.rect.x = b.pos[0]

            # test y collision
            a.rect.y = a_fut_pos[1]
            b.rect.y = b_fut_pos[1]
            would_hit_y = pygame.sprite.collide_rect(a,b) 

            # restore y component
            a.rect.x = a.pos[0]
            b.rect.x = b.pos[0]

            if would_hit_x:
                left, right = [p[-1] for p in sorted(((a.rect.x, 0, a), (b.rect.x, 1, b)))]
                #right.pos[0] = left.rect.right # old method, bias right
                #right.rect.x = right.pos[0]    # old method bias right

                # dis-intersect boxes along x-axis, evenly
                overlap_x = left.rect.right - right.rect.left
                right.pos[0] += overlap_x/2 + 1
                left.pos[0] -= overlap_x/2 - 1

                left.rect.x = left.pos[0]
                right.rect.x = right.pos[0]


            if would_hit_y and False:
                top, bot = [p[-1] for p in sorted(((a.rect.y, 0, a), (b.rect.y, 1, b)))]
                top.pos[1] = bot.rect.top + top.height
                top.rect.y = top.pos[1]

            if would_hit_x or would_hit_y: 
                #print('hit axes (x,y) = (%d,%d)'%(would_hit_x,would_hit_y))
                # velocity transfer
                a_vel = a.vel.copy()
                a.vel = b.vel.copy() #/ 1
                b.vel = a_vel #/ 1 # todo.. testing
                #for p in (a,b):
                #    if p.jumping:
                #        p.vel[1] += 500

            
    def check_if_hit_other_players(self):
        others = self.group - {self} #- set(Player.collisions) 
        hits = pygame.sprite.spritecollide(self, others, dokill=False)
        for other_player in hits:
            if other_player.getting_hit: # hit player is already in a state of being hit
                # why does this happen?
                # a player is colling but has already collided in a
                # previous frame and so _should_ be 'recoiling'
                
                #Player.collisions[hit] = self
                #Player.collisions = { }#.pop(other_player)
                continue

            # record keeping
            Player.collisions[self] = other_player
            other_player.getting_hit = True

            # velocity transfer
            other_player.vel += self.vel #/ 1
            self.vel -= self.vel #/ 1 # todo.. testing
            print('%d hit %d' % (self.player_id, other_player.player_id))

    def fetch_network_input(self):
        if self.client:
            record = self.client.get_record()
            if record:
                # TODO: give receipt confirmation and message redundancy in network layer
                # until confirmed that record seq_no was received
                if (record.seq_no != self.last_seq_no + 1):
                    self.dropped_packets += 1
                    rate = self.dropped_packets / record.seq_no
                    print('DROPPED PACKET', self.last_seq_no + 1, f'DROP RATE:{rate:.3%}')

                self.last_seq_no = record.seq_no
                self.acc_x_input = rescale(record.roll, mn=-128, mx=127, a=-1, b=1)
                self.jump_pressed = record.jump_pressed()
                self.slow_mo_pressed = record.slow_mo_pressed()
                self.slow_mo_exit_pressed = record.slow_mo_exit_pressed()
                self.dy = record.dy

    def update(self, dt): ## Player
        self.frame += 1

        self.jump_pressed = False
        self.slow_mo_pressed = False
        self.slow_mo_exit_pressed = False

        self.fetch_network_input()

        debug_msg = []
        debug_msg.append(f'P{self.player_id}:')
        #stat = Player.active_slowmo.player_id if Player.active_slowmo else 'None'
        #debug_msg.append(f'Slow Mo: {stat}, presses: {self.slowmo_triggers}, '\
        #f'enters: {self.slowmo_enters}, exits: {self.slowmo_exits}, '\
        #f'misfires: {self.slowmo_triggers-(self.slowmo_enters+self.slowmo_exits)}')

        ## Only allow one player to slow down time at once
        if self.slow_mo_pressed and self.slow_mo_exit_pressed:
            # Too quick, cancel attempt if game is normal speed, 
            # exit slowmo if already started
            # If game is normal speed, this will work fine,
            # resulting in a quick projectile fire.
            #if game.slow_mo:
            #    slow_mo_pressed = False  # only let the slow-mo exit go thru
            # lets just see if this works
            print('player sent both slow-mo flags in one network frame')

        if self.slow_mo_pressed:
            self.slowmo_triggers += 1
            if not game.slow_mo and Player.active_slowmo is None:
                # Only allow slow-mo if it's not engaged by any player.
                game.enter_slow_mo(self)

        if self.slow_mo_exit_pressed:
            self.slowmo_triggers += 1
            if game.slow_mo and Player.active_slowmo is self:
                # Only allow slow-mo exit if player is the one who
                # triggered current slow-mo.
                game.exit_slow_mo(self)

        if self.jump_pressed and not self.jumping:
            print('jump pressed') 
            self.vel[1] -= 1000
            self.jumping = True

        ## Set acceleration due to player input
        self.acc[0] = self.acc_x_input*self.MAX_ACC

        ## Add environmental forces (e.g. wind)
        for force in self.environ_forces.values():
            self.acc += force 

        ## Apply gravity 
        self.acc[1] = 3000 # assuming 1m = ~300px and gravity @ -10 m/s


        self.vel[1] += self.acc[1] * dt
        self.vel[1] = max(-self.MAX_VEL, min(self.vel[1], self.MAX_VEL))
        self.pos[1] += self.vel[1] * dt  # _subtract_ y pos due to flipped y-axis

        self.check_collision_with_ground()

        ## Update player physics
        self.vel[0] += self.acc[0] * dt
        self.vel[0] *= self.friction
        self.vel[0] = max(-self.MAX_VEL, min(self.vel[0], self.MAX_VEL))
        self.pos[0] += self.vel[0] * dt

        #self.pos += game.cam.pos # TESTING
        self.update_pos()

        ## Select the correct sprite image to display
        if self.acc[0] > 0:
            self.direction = 1
        else:
            self.direction = -1

        prev_phase = self.phase 
        phase_index = int(rescale(abs(self.vel[0]), mn=0, mx=self.MAX_VEL, 
                                                    a=0, b=self.n_phases))
        self.phase = self.phase_map[phase_index]

        self.display_image = self.images[self.direction][self.phase]

        if self.getting_hit:
            self.display_image = self.hit_im

        ## Control sounds and animations triggered by player state
        if self.phase != prev_phase:
            if self.phase == '5':
                self.sound.start_looped('sliding')
                SnowPlume(self.pos, self)
            elif prev_phase == '5':
                self.sound.stop_looped('sliding', fade_ms=200)
                SnowPlume.deactivate(self) 

        ## Camera processing
        #self.pos += game.cam.pos

        ## Apply horizontal boundary conditions
        if self.pos[0] > WIDTH:
            self.pos[0] = -self.width #640 - self.center_offset[0]
        elif self.pos[0] < -self.width:
            self.pos[0] = WIDTH # - self.center_offset[0]

        ## Record position history for ghosting effect
        self.pos_history.append(self.pos.copy())

        debug_msg.append(f'acc: [{self.acc[0]:>8.1f}, {self.acc[1]:>8.1f}]')
        debug_msg.append(f'vel: [{self.vel[0]:>8.1f}, {self.vel[1]:>8.1f}]')

        ## Collision checks
        self.check_drop_collisions() 
        #if self.player_id == 1:
        #    self.check_if_hit_other_players()

        if self.recoil_cooldown_frames > 0:
            self.recoil_cooldown_frames -= 1

        if not any(Player.collisions):
            self.getting_hit = False
        else:
            print('colliding:', len(Player.collisions))

        if True or game.debug_rects:
            game.print(" ".join(debug_msg), self.player_id)


    def draw(self, screen):
        # draw path trace trail (snow clumps)
        n = len(self.pos_history)-1
        r = 32 # random noise bound
        if self.jumping and -self.vel[1] > self.MAX_VEL / 4:# or self.vel[0]==self.MAX_VEL:
            for i in range(n):
                if 20 < self.pos_history[i][0] < WIDTH - 20 and random.randint(0,1)==1:
                    pygame.draw.circle(screen, pygame.Color(255,255,255).lerp(
                        BG_COLOR,(1-i/n)),
                        self.pos_history[i] - self.center_offset +
                        (random.randint(0,r) - r/2, random.randint(0,r) - r/2),
                        random.randint(1,7), width=0)#random.randint(0,1))
                        #self.pos_history[i+1] + self.center_offset)

        # draw ghost trail
        ngf = self.n_ghost_frames
        if self.frame > ngf:
            for i in range(ngf):
                screen.blit(self.ghost_images[self.direction][self.phase][i],
                            self.pos_history[n-ngf+i+1] - self.center_offset)
        # draw sprite
        screen.blit(self.display_image, self.rect.topleft + game.cam.offset)

        # draw reflection in ice
        max_refl_height = 200
        dist_above_ice = game.ice.mid - self.rect.bottom
        intensity = rescale(dist_above_ice, mn=0, mx=max_refl_height, a=0, b=9)
        refl_im = self.get_reflection_image(9 - clamp(int(intensity), mn=0, mx=9))
        screen.blit(refl_im, (self.rect.left, game.ice.mid) + game.cam.offset)
        

        if game.debug_rects:
            pygame.draw.rect(screen, WHITE, self.rect, width=1)
            tangent = 40*normalize((self.vel))
            pygame.draw.line(screen, PURPLE, self.pos, self.pos + tangent, width=2) 

        #print('vel:', self.vel, 'pos:', self.pos)

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
    ground_level = 80
    top = HEIGHT - ground_level #400
    mid = top + 32
    group = set()
    def __init__(self):
        self.group.add(self)
        self.color = pygame.Color(232,238,252,255)
        self.image = vertical_gradient((WIDTH+100,80 +20), BG_COLOR+(255,), self.color)
        self.pos = np.array([-50,self.top], dtype=float)
        self.rect = self.image.get_rect()

    def update(self, dt):
        pass
        #self.pos += game.cam.pos # TESTING

    def draw(self, screen):
        screen.blit(self.image, self.pos + game.cam.offset) 

class Sound:
    maj = [0,2,4,5,7,9,11]
    maj = maj + [i+12 for i in maj]
    low = 52
    SOUNDS_PATH = os.path.join('assets','sound')
    #EFFECTS_PATH = os.path.join(SOUNDS_PATH, 'effects')
    #order = ['ow','shit','fuck','balls','christ']
    instruments = {1:'guitar', 2:'bass'}
    note_orderings = {'guitar': [str(52+i) for i in maj],
                    'bass':[str(52+i) for i in maj]}
    effects = {'arrow':'laser_gun', 'slowmo':'enter_slowmo_airy',
                'intercepted':'intercepted_robo'}
    congrats = ['nice_robo', 'golden_robo',]

    # todo: dynamic sample triggering by song key, and bass follows guitar chord last
    #       hit by player 1, for example...
    def __init__(self, asset_packs, instrument=None, is_environment=False):
        self.instrument = instrument
        if instrument:
            self.order = self.note_orderings[self.instruments[instrument]]
            self.n_sounds = len(self.order)
            print('sounds registery:', asset_packs, instrument)
        self.asset_packs = asset_packs
        self._load_sounds()
        self.index = 0
        self.is_combo = False
        self.last_hit_time = 0
        if is_environment:
            self._start_bg_noise()

    def _start_bg_noise(self):
        self.sounds['static'].play(loops=-1)
        self.looping['static'] = self.sounds['static']

    def _load_sounds(self):
        self.sounds = { }
        self.looping = { }
        self.one_shots = { }
        for asset_pack in self.asset_packs:
            for path in os.scandir(os.path.join(self.SOUNDS_PATH, asset_pack)):
                name = path.name.split('.')[0]
                assert(name not in self.sounds) # no accidental over-riding
                sound = pygame.mixer.Sound(path.path)
                self.sounds[name] = sound

    def register_miss(self):
        self.is_combo = False
        self.index = 0

    def start_looped(self, name):
        snd = self.sounds[name]
        snd.play(loops=-1)
        self.looping[name] = snd

    def stop_looped(self, name, fade_ms=0):
        snd = self.looping.get(name)
        if snd:
            snd.fadeout(fade_ms)

    def start_fx(self, name):
        return self.start_one_shot(self.effects[name])

    def stop_fx(self, name, fade_ms=0):
        self.stop_one_shot(self.effects[name], fade_ms)

    def stop_sound(self, snd, fade_ms=0):
        """
        Use this to stop sounds, user sends playing sound back in.
        """
        snd.fadeout(fade_ms)

    def play_congrats(self):
        name = random.choice(self.congrats)
        self.start_one_shot(name)

    def start_one_shot(self, name):
        snd = self.sounds[name]
        snd.play()
        self.one_shots[name] = snd
        return snd

    def stop_one_shot(self, name, fade_ms=0):
        snd = self.one_shots.get(name)
        if snd:
            snd.fadeout(fade_ms)
        return snd

    def play_next(self):
        self.sounds[self.order[self.index]].play()
        perc = self.sounds.get('percussion')
        if perc:
            perc.play()
        self.index = (self.index + 1) % self.n_sounds
        self.is_combo = True



class Wind:
    def __init__(self):
        self.blowing = False
        self.frame = 0
        self.direction = 1
        self.strength = 60*3#1
        self.frequency = 10 # average seconds between gusts of wind
        self.force = np.array([0,0])

    def start(self, n_frames=60*3):
        self.n_frames = n_frames
        self.frame = 0
        self.direction = 1 if random.random() > 0.5 else -1
        #self.force = np.array([self.direction*self.strength, 0])
        #self.force = np.array([0,0]) # do sprite references to this get cleared here?
                                     # or do they still point the same memory location
        self.force[0] = 0
        self.blowing = True

    def update(self):
        if self.blowing:
            self.frame += 1
            if self.frame == self.n_frames:
                self.blowing = False
            # ease force up and down with sin curve
            #self.cycle_point = math.sin(math.pi*self.frame/self.n_frames)
            self.force[0] = self.direction*self.strength*math.sin(
                                            math.pi*self.frame/self.n_frames)
        else:
            self.chance()

    def chance(self):
        if random.randint(0,60*self.frequency) == 50:
            self.start()
            print('wind started blowing')

def close_network_connections():
  for player in Player.group:
      player.remote.set()
  netcon.shutdown()

def quit():
  close_network_connections()
  game.music_stream.shutdown()
  pygame.quit() 
  sys.exit() # Not including this line crashes the script on Windows. Possibly

class Camera:
    def __init__(self, game):
        self.game = game
        self.pos = np.array((0,0), dtype=float)
        self.dolly = np.array((0,0), dtype=float)
        self.offset = np.array((0,0), dtype=float)
        self.impact = False

    def reset(self):
        self.pos = np.array((0,0))
        self.offset = np.array((0,0))

    def shake(self):
        amount = self.shake_amount
        self.offset[0] = random.randint(0,amount) - amount/2
        self.offset[1] = random.randint(0,amount) - amount/2

    def start_shake(self, frames=20, amount=8):
        self.impact = True
        self.impact_frames = frames
        self.shake_amount = amount

    def clear_shake(self):
        self.impact = False
        self.offset = np.array((0,0))

    def update(self, dt):
        if self.impact:
            if self.impact_frames > 0:
                self.shake()
                self.impact_frames -= 1
                # todo: maybe taper the shake amount every frame
            else:
                self.clear_shake()
        self.dolly[0] = 10*math.sin((1/16)*2*np.pi*game.t/1000)
        self.dolly[1] = 5*math.sin((1/8)*2*np.pi*game.t/1000)
        self.pos = self.dolly #+ self.offset # dont mutate pos with offset TODO


class Game:
    DEBUG_ROW_HEIGHT = 18
    MAX_SLOW_FACTOR = 10
    MAX_SLOW_MO_TIME = 2000

    def __init__(self, debug_rects=False):
        self.debug_rects = debug_rects
        self._init_pygame()
        self._init_sound()
        self._init_sprite_configs()
        self.slow_mo = False
        self.slow_factor = self.MAX_SLOW_FACTOR
        self.elapsed_slow_mo_time = 0
        self.display_dt = 0
        self.t = 0
        self.stalled = False
        self.stall_time_left = 0
        self.cam = Camera(self)
        font = pygame.font.match_font(('menlo', pygame.font.get_default_font()))
        self.font = pygame.font.Font(font, 20)
        self.debug_surfs = { }
        self.music_stream = WaveStream('astley.wav', segment_dur=0.02,
                                        stretch=1.5, overwrite=True,
                                        aim_for_even_chunks=True)
        pygame.event.set_allowed([QUIT, KEYDOWN, KEYUP,
                                  self.music_stream.chunk_end_event])

    def _init_pygame(self):
        pygame.mixer.pre_init(44100, size=-16, channels=2, buffer=32)
        pygame.init()
        pygame.mixer.init()
        pygame.mixer.set_num_channels(8)

    def _init_sound(self):
        self.environ_sound = Sound(asset_packs=['environment'], is_environment=True)
        self.sound_fx = Sound(asset_packs=['effects'], is_environment=False)

    def _init_sprite_configs(self):
        Drop._init_sectors()

    def display_fps(self):
        self.print(f'FPS: {self.fps_clock.get_fps():.0f}', 0)

    def print(self, text, row):
        self.debug_surfs[row] = self.font.render(text, True, (255,255,255))

    def stall(self, ms):
        self.stalled = True
        self.stall_time_left = ms

    def resume(self):
        self.stalled = False
        self.stall_time_left = 0
    
    def enter_slow_mo(self, player):
        self.music_stream.set_rate(2)
        self.slow_factor = self.MAX_SLOW_FACTOR
        self.elapsed_slow_mo_time = 0
        self.slow_mo = True
        player.slowmo_enters += 1
        Player.active_slowmo = player
        player.arrow = Arrow(player)
        player.slowmo_snd = game.sound_fx.start_fx('slowmo')

    def exit_slow_mo(self, player):
        self.music_stream.set_rate(1)
        self.slow_mo = False
        player.slowmo_exits += 1
        Player.active_slowmo = None
        player.arrow.fire()
        self.sound_fx.stop_fx('slowmo', fade_ms=20)
        self.sound_fx.start_fx('arrow')

    def update(self, dt):
      """
      Game update phase.
      """
      ## Events
      #print('music q:', game.music_stream.channel.get_queue())
      for event in pygame.event.get():
        if event.type == QUIT:
            quit()
        #elif event.type == game.music_stream.chunk_end_event:
        #    game.music_stream.queue_next()
        elif event.type == pygame.KEYDOWN:
            if event.key == K_q:
                quit()
            elif event.key == K_s:
                #player1.left = True
                pass
            elif event.key == K_f:
                #player1.right = True
                pass
        elif event.type == pygame.KEYUP:
            if event.key == K_k:
                #self.slow_mo = True
                #Arrow.power += 5
                game.music_stream.set_rate(1)
            elif event.key == K_j:
                #Arrow.power -= 5
                #self.slow_mo = False
                game.music_stream.set_rate(2)

      pygame.event.pump()

      ## Update 
      self.display_fps()

      if self.stalled and self.stall_time_left > 0:
          self.stall_time_left -= dt
          return 
      else:
          self.resume()

      self.t += dt
      self.display_dt = dt
      if self.slow_mo:
          max_slomo_t = self.MAX_SLOW_MO_TIME
          #if self.slow_mo_exit_signal:
          #    time_left = max(0, max_slomo_t - self.elapsed_slow_mo_time)
          #    self.slow_factor = lerp_quad(self.elapsed_slow_mo_time, t_max=max_slomo_t)  

          self.elapsed_slow_mo_time += dt
          self.slow_factor = lerp_quad(self.elapsed_slow_mo_time, t_max=max_slomo_t)  
          self.display_dt = dt / clamp(self.slow_factor, mn=1, mx=10)

          if self.elapsed_slow_mo_time > self.MAX_SLOW_MO_TIME:
              game.exit_slow_mo(Player.active_slowmo)

      self.display_dt /= 1000         # convert to seconds


      self.cam.update(dt)

      self.wind.update()

      Player.collisions = { }
      Player.check_for_collisions_between_players()

      for ice in Ice.group:
          ice.update(self.display_dt)

      for player in Player.group:
          player.update(self.display_dt)
          
      for bubble in PlayerBubble.group:
          bubble.update(self.display_dt)

      if len(Drop.group) < 8:
          Drop(y=random.randint(-200, -50))

      for drop in Drop.group:
          drop.update(self.display_dt)

      for arrow in Arrow.group:
          arrow.update(self.display_dt)

      for expl in Explosion.group:
          expl.update(self.display_dt)

    def draw(self, screen):
      """
      Game draw phase.
      """
      screen.fill(BG_COLOR) # Fill the screen with black.

      for ice in Ice.group:
          ice.draw(screen)

      for player in Player.group:
          player.draw(screen)

      for drop in Drop.group:
          drop.draw(screen)

      for bubble in PlayerBubble.group:
          bubble.draw(screen)

      for arrow in Arrow.group:
          arrow.draw(screen)

      for expl in Explosion.group:
          expl.draw(screen)

      self.draw_debug_info(screen) 
      pygame.display.flip()

    def draw_debug_info(self, screen):
        for row, surf in self.debug_surfs.items():
            screen.blit(surf, (0, row*self.DEBUG_ROW_HEIGHT))

    def run(self):
      self.fps = 60.0
      self.fps_clock = pygame.time.Clock()
      
      width, height = WIDTH, HEIGHT
      screen = pygame.display.set_mode((width, height), pygame.DOUBLEBUF, 32)
      self.screen = screen
      
      self.wind = Wind()
      self.ice = Ice()

      player1 = Player(pos_x=100)
      player2 = Player(pos_x=WIDTH - 100, skin='hat_red')


      # store a copy of blank bg image
      self.bg_image = screen.copy()
      self.bg_image.fill(BG_COLOR)
      self.ice.draw(self.bg_image)

      self.music_stream.play_threaded()
      dt = 1/self.fps 
      while True:
        self.update(dt)
        self.draw(screen)
        dt = self.fps_clock.tick(self.fps)
      #  print('fps:', fpsClock.get_fps())

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def flip_y(v):
    return np.array((v[0], -v[1]))

def rescale(x, mn=-math.pi/2, mx=math.pi/2, a=0, b=WIDTH):
    return a + ((x - mn)*(b-a)) / ( mx - mn)

def clamp(x, mn=-1, mx=1):
    return min(mx, max(x, mn))

def lerp_quad(t, t_max=1, a=1, b=10):
    return a + (b-a)*(1 - math.pow(t/t_max, 2))
 
netcon = network.NetworkConnection()
netcon.listen()

def get_cli_args():
    args = sys.argv[1:]
    if len(args) > 0:
        if args[0].lower() == 'debug':
            args[0] = True
    return args

game = Game(*get_cli_args())
game.run()
