import pygame

class static_obs(pygame.sprite.Sprite):
    def __init__(self, sizeX, sizeY, sizeObs):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((sizeObs[0], sizeObs[1]))
        self.image.fill((0, 0, 0))
        self.rect = self.image.get_rect()
        self.rect.center = [sizeX, sizeY]
        self.mask = pygame.mask.from_surface(self.image)

