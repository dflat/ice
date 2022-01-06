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
import verts

BG_COLOR = (68,52,86)#(213,221,239)#(76,57,79)#(240,240,255)
RED = (255,0,0)
GREY = pygame.Color(100,100,100)#(100,)*3
PLAYER_COLOR = pygame.Color(193,94,152)
WIDTH = 1200#640
HEIGHT = 480
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
    offset = np.array([12, 32])
    flipped_offset = np.array([-offset[0], offset[1]])
    active = { }
    frame_seq = list(range(n_frames)) + list(reversed(range(n_frames)))[1:-1]

    def __init__(self, pos, player):
        super().__init__(pos)
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

    def __init__(self, pos):
        super().__init__(pos)
        self.offset = np.array([6*random.random(), 24*random.random()])

    def draw(self, screen):
        screen.blit(self.frames[self.frame_no], self.pos + self.offset)

class Drop(pygame.sprite.Sprite):
    group = pygame.sprite.Group()
    COLORS = [pygame.Color(150, 188, 222), pygame.Color(161, 206, 229),
                pygame.Color(169, 217, 231), pygame.Color(187, 228, 233),
                pygame.Color(216, 233, 236), pygame.Color(174, 220, 220)]

    _image_cache = { }

    def __init__(self, x=0, y=0, height=48, width=16):
        super().__init__()
        self.group.add(self) 
        self.width = width
        self.height = height
        self._color = random.choice(self.COLORS)
        self.color = self._color
        self.image0 = self._load_image(self.width, self.height, self.color)
        self.image = self.image0.copy()
        self.rect = self.image.get_rect()

        self.n_ghost_frames = 0 #TODO bug with ghost frames
        self._get_ghost_frames()
        self.center_offset = np.array([self.width//2,self.height//2], dtype=float)
        self.pos = np.array([x,y], dtype=float)
        self.vel = np.array([0,0], dtype=float)
        self.gravity = random.randint(10,40)
        self.acc = np.array([0,self.gravity], dtype=float)
        self.t = 0
        self.pos_history = deque(maxlen=self.n_ghost_frames)
        self.frame = 0
        self.environ_forces = defaultdict(lambda: np.array([0,0], dtype=float))
        self.twinkle_freq = random.randint(60*1, 60*2)
        self.max_rotation = random.randint(10, 40)

    def _load_image(self, w, h, color):
        im = self._image_cache.get((w,h,tuple(color)))
        if not im: 
            w, h = w/2, h/2
            im = pygame.Surface((w,h), pygame.SRCALPHA, 32).convert_alpha()
            self.vertices = [(0, h/7), ((w-1)/3, 1), ((2/3)*(w-1),1),
                                (w-1, h/8), ((w-1)/2, h-1)]
            pygame.draw.polygon(im, self.color, self.vertices)
            shading_verts = [((2/3)*(w-1),1),
                                (w-1, h/8), ((w-1)/2, h-1)]
            pygame.draw.polygon(im, self.color.lerp((255,255,255), .25), shading_verts)
            im = pygame.transform.scale2x(im)
            self._image_cache[(w,h,tuple(color))] = im
        return im
        #self.image0 = im
        #self.image = im.copy()

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

    def update(self, dt): # Drop
        self.frame += 1

        # check if off-screen
        if self.pos[1] > HEIGHT + self.height:
            #self.sound.register_miss()
            return self.kill()

        # environmental forces
        for force in self.environ_forces.values():
            self.acc += force

        # physics simulation
        dt /= 1000
        if game.slow_mo:
            dt /= 10

        self.vel += self.acc*dt
        self.vel[0] *= .90 # horizontal air resistance
        self.pos += self.vel*dt
        self.rect.x = self.pos[0]
        self.rect.y = self.pos[1]

        # remember position history
        self.pos_history.append(self.pos.copy())

        # special animations
        if self.frame % self.twinkle_freq == 0:
            Twinkle(self.pos)# + np.array([0, self.height*random.random()]))

    def draw(self, screen):
        if self.frame > self.n_ghost_frames:
            for i in range(self.n_ghost_frames):
                screen.blit(self.ghost_images[i], self.pos_history[i])
        screen.blit(self.image, self.pos + game.cam.pos)

    def _get_ghost_frames(self):
        n = self.n_ghost_frames
        self.ghost_images = []
        shift = .1 if n < 10 else 1/(n)
        self.ghost_colors = [self.color.lerp(BG_COLOR, 1-(i*shift)) for i in range(n)]
        for i in range(n):
            im = self.image.copy()
            pygame.draw.polygon(im, self.ghost_colors[i], self.vertices)
            self.ghost_images.append(im)

class Arrow(Drop): #pygame.sprite.Sprite):
    group = pygame.sprite.Group()

    def __init__(self, player):
        super().__init__()
        self.group.add(self) 
        self.player = player
        self.theta = 0
        self.color = pygame.Color(255,255,255).lerp(BG_COLOR, .5) # DUPE color attr
        self.inner_r = 80
        #self.outer_box = player.rect.inflate(0,0)

        self.ring_rect = pygame.Rect(player.rect.center, (self.inner_r*2, self.inner_r*2))
        self.arch_rect = pygame.Rect(player.rect.center, (self.inner_r*2 + 10, self.inner_r*2))
        
        # Test initializing these attributes here to allow projectile
        # to fire on first frame of existence, before a single update takes place
        self.aim_vector = np.array([self.player.direction*math.cos(self.theta),
                                            math.sin(self.theta)])
        self.tip_offset = self.inner_r*self.aim_vector
        self.aim_center = self.ring_rect.center + np.array(self.tip_offset)

        self.bullet_width = 16
        self.bullet_height = 48
        self._color = random.choice(self.COLORS)
        self.color = self._color
        self.image0 = self._load_image(self.bullet_width, self.bullet_height, self.color)
        self.image = self.image0.copy()
        self.rect = self.image.get_rect()

        self.rect.center = self.aim_center #.copy() # TEST

        self.offset = np.array((-8,-24))#-self.rect.x, -self.rect.y))
        self.pos_history = deque(maxlen=60)
        self.gravity = 20*60
        self.speed = 30*60
        self.pos = np.array(self.rect.center, dtype='float') # testing...
        self.frame = 0
        self.fired = False

    def update_pos(self):
        self.rect.x = self.pos[0]
        self.rect.y = self.pos[1]

    def update(self, dt):
        dt /= 1000
        if self.fired:
            self.vel[1] += self.gravity*dt
            self.rect.x += self.vel[0]*dt
            self.rect.y += self.vel[1]*dt
            self.pos += self.vel*dt
            pos = self.rect.center #self.aim_center + self.rect.topleft + self.offset
            print(f'dgrav:{self.gravity*dt:.4f}, rect.y:{self.rect.y}',
                    f'self.pos_y={self.pos[1]:.2f}', end='\r')
            #self.pos_history.append(self.rect.center)
            self.pos_history.append(pos)
            if pos[1] > HEIGHT + 100 or pos[0] > WIDTH + 100 or pos[0] < -100:
                self.kill()
            self.collision_check()
            # todo: rotate to velocity tangent, fix gravity in slowmo
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
            self.rect.center = self.aim_center #.copy() # testingh
            self.rotate(90*self.player.direction -
                    self.player.direction*self.theta*(180/math.pi))
        

    def collision_check(self):
        for player in Player.group - {self.player}:
            if player.rect.collidepoint(self.rect.center):
                #player.take_damage()
                self.kill()
                ProjectileExplosion(self.rect.center)
                game.cam.start_shake()

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

    def draw_trace(self, screen):
        offset = self.aim_center + self.offset
        n = len(self.pos_history)-1
        for i in range(n):
            if 20 < self.pos_history[i][0] < WIDTH - 20:
                pygame.draw.line(screen, pygame.Color(255,255,255).lerp(BG_COLOR,(1-i/n)),
                                 self.pos_history[i], self.pos_history[i+1])

def assemble_image(surf, obj_filename, color_map):
    path = os.path.join(OBJ_PATH, obj_filename)
    parts = verts.parse_obj_as_dict(path)
    for key in color_map: 
        pygame.draw.polygon(surf, color_map[key], parts[key][0])
    return surf

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
    color_map = {'base':pygame.Color(0,0,0), 'belly':pygame.Color(240,240,240),
                'feet':pygame.Color(235,191,73), 'beak': pygame.Color(235,191,73),
                'eyeball':pygame.Color(255,255,255)
                }
    def __init__(self, color=PLAYER_COLOR, width=64, height=32, pos_x=None, skin=None):
        super().__init__()
        self._register_new_player()

        self.sound = Sound(asset_packs = self.sound_packs[self.player_id],
                            instrument = self.player_id)
        self.group.add(self) 
        self.width = width*2
        self.height = height*2
        self.skin = skin
        self._init_images()
        self.image = self.images[-1]['1']
        self.rect = self.image.get_rect() 

        self._color = color
        self.color = color

        self.n_ghost_frames = 2
        self._get_ghost_frames()

        self.pos = np.array([0,Ice.top - self.height/2], dtype=float)
        if pos_x:
            self.pos[0] = pos_x
        self.center_offset = np.array([self.width//2,self.height//2], dtype=float)
        self.t = 0
        self.vel = np.array([0,0], dtype=float)
        self.acc = np.array([0,0], dtype=float)
        self.MAX_VEL = 640*2
        self.MAX_ACC = 1600*2
        self.left = False
        self.right = False
        self.pos_history = deque(maxlen=self.n_ghost_frames+10)
        self.packet_history = deque(maxlen=4)
        self.missed = 0
        self.x = 0
        self.frame = 0
        self.friction = 0.99
        self.environ_forces = defaultdict(lambda: np.array([0,0], dtype=float))
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
        self._connect_to_network()

    def _connect_to_network(self):
        self.client = None 
        self.remote = threading.Event()
        t = threading.Thread(target=netcon.link_player, args=(self,self.remote))
        t.start()

    def establish_link(self, client): # will be set by a thread in netcon class
        self.client = client

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
            flipped = pygame.transform.flip(im,True,False)
            self.images[-1][phase] = im 
            self.images[1][phase] = flipped
        self.hitbox = pygame.Surface(im.get_size())
        self.hitbox.fill((255,0,0))

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
        for phase in self.phases:
            for i in range(n):
                im = self.images[-1][phase].copy()
                im.set_alpha(self.ghost_alphas[i])
                self.ghost_images[-1][phase].append(im)
                self.ghost_images[1][phase].append(pygame.transform.flip(im,True,False)) 

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
             

    def update_pos(self):
        self.rect.x = self.pos[0]
        self.rect.y = self.pos[1]

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

    def update(self, dt): ## player
        self.dir = 1*self.right - 1*self.left
        self.t += dt
        dt_in_ms = dt
        self.frame += 1
        dt /= 1000
        if game.slow_mo:
            dt /= 10

        ## Fetch network control data
        jump_pressed = False
        slow_mo_pressed = False
        slow_mo_exit_pressed = False
        if self.client:
            record = self.client.get_record()  #self.netcon.get_record()
            if record:
                #assert(record.seq_no == self.last_seq_no + 1)
                # TODO: give receipt confirmation and message redundancy in network layer
                # until confirmed that record seq_no was received
                if (record.seq_no != self.last_seq_no + 1):
                    self.dropped_packets += 1
                    rate = self.dropped_packets / record.seq_no
                    print('DROPPED PACKET', self.last_seq_no + 1, f'DROP RATE:{rate:.3%}')
                self.last_seq_no = record.seq_no
                self.x = rescale(record.roll, mn=-128, mx=127, a=-1, b=1)
                jump_pressed = record.jump_pressed()
                slow_mo_pressed = record.slow_mo_pressed()
                slow_mo_exit_pressed = record.slow_mo_exit_pressed()
                self.dy = record.dy
                #print(record)
                #print('dy', self.dy)

        stat = Player.active_slowmo.player_id if Player.active_slowmo else 'None'
        msg = f'Slow Mo: {stat}, presses: {self.slowmo_triggers}, '\
        f'enters: {self.slowmo_enters}, exits: {self.slowmo_exits}, '\
        f'misfires: {self.slowmo_triggers-(self.slowmo_enters+self.slowmo_exits)}'
        game.print(msg, self.player_id)

        ## Only allow one player to slow down time at once
        if slow_mo_pressed and slow_mo_exit_pressed:
            # Too quick, cancel attempt if game is normal speed, 
            # exit slowmo if already started
            # If game is normal speed, this will work fine,
            # resulting in a quick projectile fire.
            #if game.slow_mo:
            #    slow_mo_pressed = False  # only let the slow-mo exit go thru
            # lets just see if this works
            print('both slow-mo flags set in one network frame')

        if slow_mo_pressed:
            self.slowmo_triggers += 1
            if not game.slow_mo and Player.active_slowmo is None:
            # Only allow slow-mo if it's not engaged by any player
                self.slowmo_enters += 1
                game.slow_mo = True
                Player.active_slowmo = self
                self.arrow = Arrow(self)

        if slow_mo_exit_pressed:
            self.slowmo_triggers += 1
            if game.slow_mo and Player.active_slowmo is self:
            # Only allow slow-mo exit if player is the one who triggered current slow-mo
                self.slowmo_exits += 1
                game.slow_mo = False
                Player.active_slowmo = None
                self.arrow.fire()

        # slow-mo timeout in case finish packet is dropped.. hack 'til fix network bug
        #if Player.active_slowmo is self:
        #    self.slow_mo_dur += dt_in_ms
        #    if self.slow_mo_dur > 10000:
        #        self.slow_mo_dur = 0
        #        Player.active_slowmo = None
        #        self.arrow.kill()
        #        game.slow_mo = False
        #        print('auto-timed out of slowmo')

        if jump_pressed and not self.jumping:
            print('jump pressed') 
            self.vel[1] += 1000
            self.jumping = True

        ## Set acceleration due to player input
        self.acc[0] = self.x*self.MAX_ACC

        ## Add environmental forces (e.g. wind)
        for force in self.environ_forces.values():
            self.acc += force 

        ## Apply gravity 
        self.acc[1] = -3000 # assuming 1m = ~300px and gravity @ -10 m/s


        self.vel[1] += self.acc[1] * dt
        self.vel[1] = max(-self.MAX_VEL, min(self.vel[1], self.MAX_VEL))
        self.pos[1] -= self.vel[1] * dt  # _subtract_ y pos due to flipped y-axis

        # TODO: fix this hack
        if self.vel[1] < 0 and self.pos[1] >= Ice.top - self.height/2:# + 2:
            self.jumping = False
            #print('stopped')
            self.vel[1] = 0

        self.pos[1] = min(self.pos[1], Ice.top - self.height/2) # dont fall thru floor

        ## Update player physics
        self.vel[0] += self.acc[0] * dt
        self.vel[0] *= self.friction
        self.vel[0] = max(-self.MAX_VEL, min(self.vel[0], self.MAX_VEL))
        self.pos[0] += self.vel[0] * dt
        self.rect.x = self.pos[0]
        self.rect.y = self.pos[1]

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
            self.display_image = self.hitbox

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


    def draw(self, screen):
        # draw path trace trail (snow clumps)
        n = len(self.pos_history)-1
        r = 32 # random noise bound
        if self.jumping and self.vel[1] > self.MAX_VEL / 4:# or self.vel[0]==self.MAX_VEL:
            for i in range(n):
                if 20 < self.pos_history[i][0] < WIDTH - 20 and random.randint(0,1)==1:
                    pygame.draw.circle(screen, pygame.Color(255,255,255).lerp(
                        BG_COLOR,(1-i/n)),
                        self.pos_history[i] + self.center_offset +
                        (random.randint(0,r) - r/2, random.randint(0,r) - r/2),
                        random.randint(1,7), width=0)#random.randint(0,1))
                        #self.pos_history[i+1] + self.center_offset)

        # draw ghost trail
        ngf = self.n_ghost_frames
        if self.frame > ngf:
            for i in range(ngf):
                screen.blit(self.ghost_images[self.direction][self.phase][i],
                                                self.pos_history[n-ngf+i+1])
        # draw sprite
        screen.blit(self.display_image, self.pos + game.cam.pos)

        # draw velocity vector representation
        #pygame.draw.line(screen, (255,0,255), self.rect.center, 
        #                            self.rect.center + 100*np.array([self.vel[0],
        #                                        -self.vel[1]])/np.linalg.norm(self.vel),
        #                                        width = 4)
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
    top = 400
    group = set()
    def __init__(self):
        self.group.add(self)
        self.color = pygame.Color(232,238,252,255)
        self.image = vertical_gradient((WIDTH+100,80 +20), BG_COLOR+(255,), self.color)
        self.pos = np.array([-50,self.top])
        self.rect = self.image.get_rect()
    def draw(self, screen):
        screen.blit(self.image, self.pos + game.cam.pos) 

class Sound:
    maj = [0,2,4,5,7,9,11]
    maj = maj + [i+12 for i in maj]
    low = 52
    SOUNDS_PATH = os.path.join('assets','sound')
    #order = ['ow','shit','fuck','balls','christ']
    instruments = {1:'guitar', 2:'bass'}
    note_orderings = {'guitar': [str(52+i) for i in maj],
                    'bass':[str(52+i) for i in maj]}
    pygame.mixer.init()
    pygame.mixer.set_num_channels(32)

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
        for asset_pack in self.asset_packs:
            for path in os.scandir(os.path.join(self.SOUNDS_PATH, asset_pack)):
                name = path.name.split('.')[0]
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

    def play_next(self):
        self.sounds[self.order[self.index]].play()
        perc = self.sounds.get('percussion')
        if perc:
            perc.play()
        self.index = (self.index + 1) % self.n_sounds
        self.is_combo = True

environ_sound = Sound(asset_packs=['environment'], is_environment=True)


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
            self.frame += 1
            if self.frame == self.n_frames:
                self.blowing = False
        else:
            self.chance()

    def chance(self):
        if random.randint(0,60*self.frequency) == 50:
            #print('starting wind')
            self.start()
wind = Wind()

def close_network_connections():
  for player in Player.group:
      player.remote.set()
  netcon.shutdown()

def quit():
  close_network_connections()
  pygame.quit() 
  sys.exit() # Not including this line crashes the script on Windows. Possibly

class Camera:
    def __init__(self, game):
        self.game = game
        self.pos = np.array((0,0))
        self.dolly = np.array((0,0))
        self.offset = np.array((0,0))
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

        self.pos = self.dolly + self.offset # dont mutate pos with offset TODO

class Game:
    DEBUG_ROW_HEIGHT = 20

    def __init__(self):
        pygame.init()
        self.slow_mo = False
        self.slow_mo_rate = 10
        self.display_dt = 0
        self.cam = Camera(self)
        font = pygame.font.get_default_font()
        self.font = pygame.font.SysFont(font, 18)
        self.debug_surfs = {1: pygame.Surface((0,0)), 2: pygame.Surface((0,0))}

    def print(self, text, row):
        self.debug_surfs[row] = self.font.render(text, True, (255,255,255))

    def update(self, dt): # game
      """
      Game update phase.
      """
      ## Events
      for event in pygame.event.get():
        if event.type == QUIT:
            quit()
        if event.type == pygame.KEYDOWN:
            if event.key == K_q:
                quit()
            elif event.key == K_s:
                #player1.left = True
                pass
            elif event.key == K_f:
                #player1.right = True
                pass
        elif event.type == pygame.KEYUP:
            if event.key == K_s:
                self.slow_mo = True
            elif event.key == K_f:
                self.slow_mo = False

      ## Update 
      self.display_dt = dt
      if self.slow_mo:
          self.display_dt = dt / 10

      if len(Drop.group) < 10:
          Drop(x=random.randint(16, WIDTH-16), y=random.randint(-500, -50))

      self.cam.update(dt)

      wind.update()

      Player.collisions = { }

      Player.check_for_collisions_between_players()
      for player in Player.group:
          # handle wind acceleration
          if wind.blowing:
              player.apply_force(wind.force*2, 'wind')
          else:
              player.clear_force('wind')

          # collision check between players
          #player.check_if_hit_other_players()

          player.update(dt)
      #Player.check_for_collisions_between_players()
          
     
      for drop in Drop.group:
          if wind.blowing:
              drop.apply_force(wind.force*0.025, 'wind')
              drop.rotate(drop.max_rotation*wind.direction*math.sin(
                                    math.pi*wind.frame/wind.n_frames))
          else:
              drop.clear_force('wind')
          drop.update(dt)

      for arrow in Arrow.group:
          arrow.update(self.display_dt)
          #arrow.update(dt)

      for expl in Explosion.group:
          expl.update(dt)

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

      for arrow in Arrow.group:
          arrow.draw(screen)

      for expl in Explosion.group:
          expl.draw(screen)

      self.draw_debug_info(screen) 
      pygame.display.flip()

    def draw_debug_info(self, screen):
        for row, surf in self.debug_surfs.items():
            screen.blit(surf, (0, (row-1)*self.DEBUG_ROW_HEIGHT))

    def run(self):
      fps = 60.0
      fpsClock = pygame.time.Clock()
      
      width, height = WIDTH, HEIGHT
      screen = pygame.display.set_mode((width, height), pygame.DOUBLEBUF, 32)
      
      player1 = Player(pos_x=100)
      player2 = Player(pos_x=WIDTH - 100, skin='hat_red')

      ice = Ice()

      dt = 1/fps 
      while True:
        self.update(dt)
        self.draw(screen)
        dt = fpsClock.tick(fps)
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
 
netcon = network.NetworkConnection()
netcon.listen()

game = Game()
game.run()
