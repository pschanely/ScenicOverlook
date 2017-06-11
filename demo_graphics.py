import collections
import random
import time
import sys

import pygame
from pygame.locals import *
import pygame.surfarray as surfarray

from scenicoverlook import viewablelist
from scenicoverlook import viewabledict

pygame.init()

W = 700
num_bars = int(sys.argv[1]) if len(sys.argv) == 2 else 50
r = random.Random()
arena_rect = Rect(0, 0, W, W)

class Bar(collections.namedtuple('bar', ['rect', 'color'])):
    def __hash__(self):
        rect = self.rect
        return hash((hash(self.color), rect.x, rect.y, rect.w, rect.h))


def random_bar():
    x=r.randint(10, W-10)
    y=r.randint(10, W-10)
    w=r.randint(50, 130)
    h=r.randint(50, 130)
    return Bar(
        rect = Rect(x-w, y-h, w, h).clip(arena_rect),
        color = tuple(r.randint(50, 190) for _ in range(3))
    )

def subtract(r1, r2):
    ret = []
    mid = r1.clip(r2)
    if r1.top < mid.top:
        ret.append(r1.clip(Rect(0,r1.top,W,mid.top-r1.top)))
    if r1.left < mid.left:
        ret.append(r1.clip(Rect(r1.left,mid.top,mid.left-r1.left,mid.height)))
    if mid.right < r1.right:
        ret.append(r1.clip(Rect(mid.right,mid.top,r1.right-mid.right,mid.height)))
    if mid.bottom < r1.bottom:
        ret.append(r1.clip(Rect(0,mid.bottom,W,r1.bottom)))
    return ret

def overlay_displays(overlay, base):
    # overlay one set of bars on top of another
    # return base.union(overlay)
    ret = []
    checkrects = [ob.rect for ob in overlay]
    base = list(base)
    overlay_rects = []
    while base:
        b = base.pop()
        idx = b.rect.collidelist(checkrects)
        if idx == -1:
            ret.append(b)
            checkrects.append(b.rect)
        else:
            base.extend(Bar(r,b.color) for r in subtract(b.rect, checkrects[idx]))
    return overlay.union(set(ret))


def to_singleton_list(item):
    return viewablelist([item])

def to_singleton_set(item):
    return set([item])

def union_sets(s1, s2):
    return s1.union(s2)

def bars_to_set(vlist):
    return vlist.map(to_singleton_set).reduce(union_sets)

def main():
    bars = viewablelist([random_bar() for _ in range(num_bars)]+[Bar(Rect(0,0,W,W), (0,0,0))])
    screen = pygame.display.set_mode((W, W), 0)
    pygame.display.set_caption('graphics demo')
    lastdisplay = set()
    baridx = r.randint(0, len(bars)-2)
    barvx, barvy, barax, baray = r.randint(0,0), r.randint(3,3), r.randint(1,1), 0
    iteration = 0
    while(True):
        iteration += 1
        if bars[baridx].rect.x > W:
            bars = bars[:baridx] + bars[baridx+1:]
            if len(bars)==1:
                break
            baridx = r.randint(0, len(bars)-2)
            barvx, barvy, barax, baray = r.randint(0,0), r.randint(3,3), r.randint(1,1), 0
        else:
            bar = bars[baridx]
            bars = bars[:baridx] + viewablelist([Bar(bar.rect.copy().move(barvx, barvy),bar.color)]) + bars[baridx+1:]
            barvx += barax
            barvy += baray
        t0 = time.time()
        nonoverlapping = bars.map(to_singleton_set).reduce(overlay_displays)
        t1 = time.time()
        #print (iteration,  ' ', t1-t0, ' ', len(nonoverlapping))
        for bar in nonoverlapping.difference(lastdisplay):
            screen.fill(bar.color, bar.rect)

        lastdisplay = nonoverlapping

        pygame.display.flip()
        pygame.event.get()
        pygame.time.wait(5)

main()
