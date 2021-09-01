import pygame
import math
import numpy as np
import random
import os


class RTK_cls(pygame.sprite.Sprite):
    def __init__(self, env, pos, player_img, range_lidar, head_velocity):
        self.dt = 0.1
        pygame.sprite.Sprite.__init__(self)
        self.x_pos = pos[0]
        self.y_pos = pos[1]

        self.img = player_img
        self.img.set_colorkey((0, 0, 0))
        self.image = self.img
        self.rect = pygame.Rect(0, 0, 40, 21)
        self.rect.center = (self.x_pos, self.y_pos)
        self.lineral_speed = 10
        self.angular_speed = 0.1
        self.last_pos = 0
        self.move_state_RTK = {
            0: [1, 0],
            1: [1, 1],
            2: [0, 1],
            3: [-1, 1],
            4: [-1, 0],
            5: [-1, -1],
            6: [0, -1],
            7: [1, -1]
        }
        self.state_angular = {
            0: 0,
            1: 45,
            2: 90,
            3: 135,
            4: 180,
            5: -135,
            6: -90,
            7: -45
        }
        self.theta = self.state_angular[random.randint(0, 7)]
        self.env = env

        self.range_lidar = range_lidar
        self.revie_lidar = (-math.pi/4, math.pi/4)
        self.angel_lidar = np.linspace(self.revie_lidar[0], self.revie_lidar[1], 90)

        self.pointLidar = ()

        self.head_angle_velocity = math.degrees(head_velocity)  # скорость поворота башни


    def update(self, action):
        self.last_pos = [self.x_pos, self.y_pos]
        moved = self.move_state_RTK[action]
        self.x_pos += moved[0] * 3  # self.env.width/100
        self.y_pos -= moved[1] * 3  # *self.env.height/100
        self.theta = self.state_angular[action]
        self.image = pygame.transform.rotozoom(self.img, self.theta, 1)
        self.image.set_colorkey((0, 0, 0))
        self.rect = self.image.get_rect(center=(self.x_pos, self.y_pos))

        n = 1+1
    def update2(self):
        ange = math.degrees(np.arctan2(-(self.env.RTK.y_pos - self.y_pos), (self.env.RTK.x_pos - self.x_pos)))
        a = ange - self.theta
        if a > 180:
            a -= 360
        if a < -180:
            a += 360
        if math.fabs(ange - self.theta) > 10:
            if a > 0:
                self.theta += self.head_angle_velocity
            else:
                self.theta -= self.head_angle_velocity
        if self.theta > 180:
            self.theta -= 360
        if self.theta < -180:
            self.theta += 360
        self.image = pygame.transform.rotozoom(self.img, self.theta, 1)
        self.image.set_colorkey((0, 0, 0))
        self.rect = self.image.get_rect(center=(self.x_pos, self.y_pos))

        n = 1+1
    def state(self):
        return self.x_pos, self.y_pos

    def change_start_pos(self, pos):
        self.x_pos = pos[0]
        self.y_pos = pos[1]
        self.rect.center = (self.x_pos, self.y_pos)
        self.image = pygame.transform.rotozoom(self.img, self.theta, 1)
        self.image.set_colorkey((0, 0, 0))
        self.rect = self.image.get_rect(center=(self.x_pos, self.y_pos))

    def sesor(self):
        data, self.pointLidar = self.sense_obstacle()

    def distance(self, obstaclePostion):
        px = (obstaclePostion[0] - self.x_pos) ** 2
        py = (obstaclePostion[1] - self.y_pos) ** 2
        return math.sqrt(px+py)

    def sense_obstacle(self):
        data = []
        points = []
        points.append((self.x_pos, self.y_pos))
        x1, y1 = self.x_pos, self.y_pos
        for angles in self.angel_lidar:
            angle = angles + math.radians(self.theta)
            x2, y2 = (x1 + self.range_lidar * math.cos(angle), y1 - self.range_lidar * math.sin(angle))
            for i in range(0, self.range_lidar+1):
                u = i / self.range_lidar
                x = int(x2 * u + x1 * (1 - u))
                y = int(y2 * u + y1 * (1 - u))
                if 0 < x < self.env.width and 0 < y < self.env.height:
                    color = self.env.map.get_at((x, y))
                    if (color[0], color[1], color[2]) == (0, 0, 0):
                        distance = self.distance((x, y))
                        data.append(distance)
                        points.append((x, y))
                        break
                    elif i == self.range_lidar:
                        distance = self.distance((x, y))
                        points.append((x, y))
                        data.append(distance)
                else:
                    distance = self.distance((x, y))
                    points.append((x, y))
                    data.append(distance)

                    n = 1+1
        return data, points
    def draw_boom(self):
        self.img = self.env.boom
        self.img.set_colorkey((255, 255, 255))
        self.image = self.img
        self.rect = pygame.Rect(0, 0, 40, 40)
        self.rect.center = (self.x_pos, self.y_pos)
        self.image = pygame.transform.rotozoom(self.img, 0, 1)
        self.image.set_colorkey((0, 0, 0))
        self.image.set_colorkey((245, 245, 245))
        self.rect = self.image.get_rect(center=(self.x_pos, self.y_pos))