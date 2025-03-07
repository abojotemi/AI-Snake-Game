from collections import deque
from itertools import islice
import sys
import pygame
class SliceableDeque(deque):
    def __getitem__(self, s: slice):
        try:
            start, stop, step = s.start or 0, s.stop or sys.maxsize, s.step or 1
        except AttributeError:  # not a slice but an int
            return super().__getitem__(s)
        try:  # normal slicing
            return islice(self, start, stop, step)
        except ValueError:  # incase of a negative slice object
            length = len(self)
            start, stop = length + start if start < 0 else start, length + stop if stop < 0 else stop
            return islice(self, start, stop, step)
        
        
CELL_SIZE = 40
CELL_NUMBER = 20
SCREEN_SIZE = CELL_SIZE * CELL_NUMBER
SCREEN = pygame.display.set_mode((SCREEN_SIZE,SCREEN_SIZE))
CLOCK = pygame.time.Clock()
SCORE = 0
head_images = {img[:-4]:pygame.image.load("./Graphics/head_" + img) for img in ["down.png", "left.png", "up.png" ,"right.png"]}
tail_images = {img[:-4]:pygame.image.load("./Graphics/tail_" + img) for img in ["down.png", "left.png", "up.png" ,"right.png"]}
turn_images = {img[:-4]:pygame.image.load("./Graphics/body_" + img) for img in ["bl.png", "br.png", "tl.png" ,"tr.png"]}
align_images = {img[:-4]:pygame.image.load("./Graphics/body_" + img) for img in ["horizontal.png", "vertical.png"]}
