import pygame
import itertools
import sys
import random
import time
from pygame.locals import *
import numpy as np

W, H = 800,600
cache = { }
BLACK = (0,0,0)
RED = (0,0,0)
WHITE = (100,)*3#(255,255,255)

class Game:
    _dirty = []
    dir = 0

def events(block):
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == KEYDOWN:
            if event.key == K_s:
                block.state.s[0] = 400
            elif event.key == K_j:
                Game.dir = np.array([-1,0])
            elif event.key == K_k:
                Game.dir = np.array([1,0])

class State:
    __slots__ = ('s','v')
    def __init__(self, s=None, v=None):
        if s is None:
            s = np.zeros(2)
        if v is None:
            v = np.zeros(2)
        self.s = s
        self.v = v
    def copy(self):
        return State(self.s.copy(), self.v.copy())
    def __mul__(self, other:float):
        return State(self.s*other, self.v*other)
    def __rmul__(self, other:float):
        return self * other
    def __add__(self, other:'State'):
        return State(self.s + other.s, self.v + other.v)
    def __sub__(self, other:'State'):
        return State(self.s - other.s, self.v - other.v)
    def __repr__(self):
        return f'State(s:{repr(self.s)}, v:{repr(self.v)})'

class Derivative:
    __slots__ = ('ds','dv')
    def __init__(self, ds, dv):
        self.ds = ds
        self.dv = dv

def acceleration(state, t):
    k = 10
    b = 1
    return -k * state.s - b*state.v

def evaluate(initial:State, t, dt, deriv:Derivative):
    state = State(initial.s + deriv.ds*dt, initial.v + deriv.dv*dt)
    return Derivative(state.v, acceleration(state, t+dt))

class Block:
    group = set()
    def __init__(self, w=40,h=40,x=400,y=0):
        self.group.add(self)
        self.image = pygame.Surface((w,h)).convert()
        self.image.fill(WHITE)
        self.eraser = self.image.copy()
        self.eraser.fill(RED)
        self.rect = self.image.get_rect()

        self.s = np.array([x,y], dtype=float) # displacement / position
        self.v = np.zeros(2)
        self.ds = np.zeros(2) # deriv. of position (velocity)
        self.dv = np.zeros(2) # deriv. of velocity (acceleration)

        self.state = State(self.s, self.v)
        self.offset = np.array([W/2, H-h])

    def acceleration(self, t):
        k = 10
        b = 1
        return -k * self.s - b*self.v

    def evaluate(self, t, dt):
        self.s += self.ds * dt
        self.v += self.dv * dt

        self.ds = self.v.copy()
        self.dv = self.acceleration(t+dt)

    def integrate(self, t, dt):
        #self.evaluate(t, dt)
        state = self.state
        a = evaluate(state, t, 0, Derivative(self.ds, self.dv))
        b = evaluate(state, t, dt*0.5, a)
        c = evaluate(state, t, dt*0.5, b)
        d = evaluate(state, t, dt, c)

        dxdt = 1/6.0 * (a.ds + 2*(b.ds + c.ds) + d.ds)
        dvdt = 1/6.0 * (a.dv + 2*(b.dv + c.dv) + d.dv)

        state.s = state.s + dxdt * dt
        state.v = state.v + dvdt * dt
        
    def update_pos(self):
        self.old_rect = self.rect.copy()
        self.old_rect.move_ip(self.offset)
        Game._dirty.append(self.old_rect)
        self.rect.center = self.state.s
        new_rect = self.rect.copy()
        new_rect.move_ip(self.offset)
        Game._dirty.append(new_rect)

    def update(self, t, dt):
        self.integrate(t,dt) 
        self.update_pos()

    def draw(self, screen):
        #screen.blit(self.eraser, self.old_rect.topleft)
        screen.blit(self.image, self.offset+self.rect.topleft)

def update(t, dt):
    for block in Block.group:
        block.update(t, dt)

def clear_dirty_rects():
   pygame.display.update(Game._dirty)
   Game._dirty = []

def draw(screen):
    screen.fill(BLACK)

    for block in Block.group:
        block.draw(screen)

    #clear_dirty_rects()
    pygame.display.flip()

def run():
    pygame.init()
    screen = pygame.display.set_mode((W, H), DOUBLEBUF, 32,vsync=1)
    screen.fill(BLACK)
    #fps_clock = pygame.time.Clock()
    dt = .01
    frame = 0
    t = 0
    block = Block()
    current_time = time.time()
    accumulator = 0
    previous_state = State()
    current_state = State()
    updates = 0
    draws = 0
    while True:
#        events(block)
#        update(t, dt)
#        draw(screen)
#        continue
        #dt /= 1000
        #t += dt
        frame += 1
        new_time = time.time()
        frame_time = new_time - current_time
        if frame_time > .25:
            frame_time = .25 # avoid sprial
        current_time = new_time
        accumulator += frame_time
        events(block)
        while accumulator >= dt:
            previous_state = block.state.copy()
            update(t, dt)
            updates += 1
            t += dt
            accumulator -= dt

        alpha = accumulator / dt
        block.state = block.state*alpha + previous_state*(1 - alpha)
        update(t, dt)
        draw(screen)
        draws += 1
        #print(f'updates: {updates}, draws: {draws} frame_t:{frame_time:.4f}', end='\n')
        print(f'diff:{previous_state - block.state}', end='\r')
        #dt = fps_clock.tick(fps)
run()
