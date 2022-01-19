import pygame, sys
from pygame.locals import *
import math
from math import cos,sin
import time
import numpy as np
from collections import deque


class Node:
    _id = 0
    def __init__(self, points, a=None, b=None):
        self.points = points
        self.rect = getAABB(points)
        self.line_segment = np.array((points[0], points[-1]))
        self.a = a
        self.b = b
        self.id = Node._id
        Node._id += 1

DEBUG_LEVELS = [4]#[0,1,2,3]
def display_tree(node, level=0):
    if node:
        if level in DEBUG_LEVELS:
            pygame.draw.rect(DISPLAYSURF, COLORS[level], node.rect, width=1)
        if level == 3:
            pygame.draw.line(DISPLAYSURF, COLORS[4], *node.line_segment, width=2)
        display_tree(node.a, level+1) 
        display_tree(node.b, level+1) 


def make_children(node, points, level=1):
    if level >= BB_LEVELS:
        return
    size = len(points) // 2
    a, b = points[:size], points[size-1:]
    node.a = Node(a) 
    node.b = Node(b) 
    make_children(node.a, a, level+1)
    make_children(node.b, b, level+1)


def intersect(A,B,C,D) -> bool:
    """
    Tests if two line segments AB and CD intersect.
    """
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)


def ccw(A,B,C) -> bool:
    """
    Tests if the points A, B, and C are arranged
    in counter-clockwise order by comparing the
    slope of line AB with the slope of line AC.
    """
    b, a = B - A     # get slope AB b=dx, a=dy -- m1 = dy/dx = a/b
    d, c = C - A     # get slope AC d=dx, c=dy -- m2 = dy/dx = c/d 
    return b*c > a*d # if (m1 < m2) => (bc > ad) by simple algebra


def getAABB(points):
    top_left = np.min(points,axis=0)
    bot_right = np.max(points,axis=0)
    return pygame.Rect(top_left, bot_right-top_left)


def geom_center(verts):
    return np.sum(verts,axis=0)/len(verts)


def get_rot_mat(theta):
    return np.array([(cos(theta), -sin(theta), 0),
                     (sin(theta), cos(theta), 0),
                     (0,0,1)])
def get_scale_mat(r):
    return np.array([(r,0,0),(0,r,0),(0,0,1)])
def get_two_to_three_mat():
    return np.array([(1,0),(0,1),(0,0)])
def get_three_to_two_mat():
    return np.array([(1,0),(0,1),(0,0)])
def set_translate(M, tx, ty):
    M[:,2] = (tx,ty,1)

## Initialize player variables
verts = np.array([(2,0,1),(4,2,1),(2,5,1),(0,2,1)], dtype=float)
HISTORY_TRAIL = 60
pos_history = deque(np.zeros((HISTORY_TRAIL,2)), maxlen=HISTORY_TRAIL)
seg_len_history = deque(maxlen=HISTORY_TRAIL-1)
#pos_history_arr = np.zeros((HISTORY_TRAIL, 2))
norm_history = deque(maxlen=HISTORY_TRAIL)
vert_history = deque(maxlen=HISTORY_TRAIL)
BB_LEVELS = 4

MAX_VEL = 10_000
center_shift = geom_center(verts) - (0,0,1)
verts -= center_shift # do this so rotation is about the mesh's geometric center
offset = np.array([100,100,0])
pos_offset = (offset + center_shift)[:2]
diamond_normal = np.array((0,1))
vel = np.zeros(2)
pos = np.array(pos_offset, dtype=float)

## Initialize game variables
WIDTH, HEIGHT = 800,600
pygame.init()
DISPLAYSURF = pygame.display.set_mode((WIDTH, HEIGHT), pygame.DOUBLEBUF, 32)

WHITE = pygame.Color(255,255,255)#(255,)*3
BLACK = pygame.Color(0,0,0)#(0,)*3
COLORS = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,0,255)]

FPS = 60
loops = 0
loops_this_sec = 0
accum_time = 0
accum_sec = 0
last_time = time.time()
fps_clock = pygame.time.Clock()
t = 0

### Main game loop
while True: 
    ## Time bookkeeping
    loops += 1
    loops_this_sec += 1
    cur_time = time.time()
    dt = cur_time - last_time
    last_time = cur_time
    accum_time += dt
    accum_sec += dt
    t = accum_time

    if accum_sec >= 1:
        print(f'{loops_this_sec/accum_sec:.1f} loops/sec')
        accum_sec = accum_sec - 1
        loops_this_sec = 0

    ## Event processing
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == KEYDOWN:
            if event.key == K_s:
                vel[0] = -MAX_VEL
            elif event.key == K_f:
                vel[0] = MAX_VEL
            elif event.key == K_e:
                vel[1] = -MAX_VEL
            elif event.key == K_d:
                vel[1] = MAX_VEL
        elif event.type == KEYUP:
            if event.key == K_s:
                vel[0] = 0
            elif event.key == K_f:
                vel[0] = 0
            elif event.key == K_e:
                vel[1] = 0
            elif event.key == K_d:
                vel[1] = 0
    DISPLAYSURF.fill(BLACK)

    ## Update world state
    mouse_pos = pygame.mouse.get_pos()
    mouse_vec = mouse_pos - pos
    norm = np.linalg.norm(mouse_vec)
    mouse_dir = mouse_vec / norm if norm else diamond_normal
    scale = 10#norm/10 
    dot_prod = diamond_normal.dot(mouse_dir)
    print(f'dot:{dot_prod:.2f}', end='\r')
    theta = np.arccos(dot_prod) 
    theta = theta if mouse_pos[0] < pos[0] else -theta

    speed = min(MAX_VEL, 5*norm)
    vel = speed*mouse_dir
    pos += vel*dt
    tx = pos[0]
    ty = pos[1]
    R = get_rot_mat(theta)
    M = get_scale_mat(scale)
    set_translate(M, tx, ty)
    A = M.dot(R)
    trans_verts = A.dot(verts.T).T
    render_verts = (trans_verts)[:,:2]
    #tail_offset = pos - render_verts[0]
    pos_history.appendleft(pos.copy())
    norm_history.appendleft(norm)
    #pos_history_arr[loops % HISTORY_TRAIL] = pos 

    ## Draw
    for i in reversed(range(min(loops-1,HISTORY_TRAIL-1))):
        color = WHITE.lerp(BLACK, i/HISTORY_TRAIL)  # 1 is pure BLACK
        pygame.draw.line(DISPLAYSURF, color, pos_history[i], pos_history[i+1],
                         #width=max(1, int(norm)//20))
                         width=max(1, int(norm_history[i])//20))
    pygame.draw.polygon(DISPLAYSURF, WHITE, render_verts)
    pygame.draw.aalines(DISPLAYSURF, WHITE, True, render_verts)

    # build bbox tree
    pos_hist_arr = np.array(pos_history)
    box_tree = Node(pos_hist_arr) # todo: use views into slices of pos_hist_arr...
    make_children(box_tree, pos_hist_arr) # ...should be significant efficiency gain
                                        # would only have to build the tree structure
                                        # once, instead of every frame...
                                        # and could just recalculate rects every frame.
                                        # as Node points would be views into pos_hist_arr

    ## Debugging rects
    display_tree(box_tree)

    pygame.display.flip()
    fps_clock.tick(FPS)
