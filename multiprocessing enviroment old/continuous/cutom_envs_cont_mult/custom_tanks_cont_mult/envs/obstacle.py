import pygame
import random

class Obstacle(pygame.sprite.Sprite):
    def __init__(self, sizeX, sizeY, sizeObs):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((random.randint(sizeObs[0], sizeObs[1]), random.randint(sizeObs[0], sizeObs[1])))
        self.image.fill((0, 0, 0))
        self.rect = self.image.get_rect()
        self.rect.center = [random.randint(0, sizeX), random.randint(0, sizeY)]
        self.mask = pygame.mask.from_surface(self.image)

