import gym
import pygame
import numpy as np

def load_pygame_image(name):
    image = pygame.image.load(name)
    return image

class ImgSprite(pygame.sprite.Sprite):
    def __init__(self, rect_pos=(5, 5, 64, 64)):
        super(ImgSprite, self).__init__()
        self.image = None
        self.rect = pygame.Rect(*rect_pos)

    def update(self, image):
        if isinstance(image, str):
            self.image = load_pygame_image(image)
        else:
            self.image = pygame.surfarray.make_surface(image)

class Render(object):
    def __init__(self, size=(320, 320)):
        pygame.init()
        self.screen = pygame.display.set_mode(size)
        self.group = pygame.sprite.Group(ImgSprite()) # the group of all sprites

    def render(self, img):
        img = np.asarray(img).transpose(1, 0, 2)
        self.group.update(img)
        self.group.draw(self.screen)
        pygame.display.flip()
        e = pygame.event.poll()

def color_interpolate(x, start_color, end_color):
    assert ( x <= 1 ) and ( x >= 0 )
    if not isinstance(start_color, np.ndarray):
        start_color = np.asarray(start_color[:3])
    if not isinstance(end_color, np.ndarray):
        end_color = np.asarray(end_color[:3])
    return np.rint( (x * end_color + (1 - x) * start_color) * 255.0 ).astype(np.uint8)
