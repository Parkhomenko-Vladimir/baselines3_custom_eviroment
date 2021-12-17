import pygame
import math
import numpy as np
import random

class RTK_cls(pygame.sprite.Sprite):
    def __init__(self, env, pos, player_img, rangelidar, velocity_head, num):
        self.dt = 0.1
        pygame.sprite.Sprite.__init__(self)
        self.id = num
        self.x_pos = pos[0]
        self.y_pos = pos[1]
        self.theta = random.uniform(-math.pi, math.pi)
        self.img = player_img
        self.img.set_colorkey((0, 0, 0))
        self.image = self.img
        self.rect = pygame.Rect(0, 0, 20, 10)
        self.rect.center = (self.x_pos, self.y_pos)
        self.lineral_speed = 10
        self.angular_speed = 0.1
        self.last_pos = 0
        self.env = env
        self.num = num
        self.state_life = False
        self.state_life2 = False
        self.range_lidar = rangelidar
        self.revie_lidar = (-math.pi / 4, math.pi / 4)
        self.num_ray = 90
        self.angel_lidar = np.linspace(self.revie_lidar[0], self.revie_lidar[1], self.num_ray)
        self.pointLidarFull = np.full((self.angel_lidar.shape[0]), self.range_lidar)
        self.pointLidar = np.zeros((self.angel_lidar.shape[0] + 2, 2))
        self.color = np.zeros(4)
        self.neighbor = 0
        self.non_stolk = False
        #self.mat_ange = np.repeat(self.angel_lidar, self.num_ray).reshape(self.num_ray*self.range_lidar)


        self.head_angle_velocity = velocity_head  #скорость поворота башни


    def update(self, action):
        if not self.state_life:
            self.last_pos = [self.x_pos, self.y_pos, self.theta]
            self.x_pos += action[0] * math.cos(self.theta) * self.dt
            self.y_pos -= action[0] * math.sin(self.theta) * self.dt
            self.theta += action[1] * self.dt
            if self.theta > math.pi:
                self.theta -= 3.14*2
            if self.theta < -math.pi:
                self.theta += 3.14*2
            if 5 > self.x_pos or self.x_pos > self.env.width - 5 or 5 > self.y_pos or self.y_pos > self.env.height - 5:
                self.edge_RTK()
            else:
                self.draw_pos()

        else:
            self.image = pygame.Surface([0, 0])
            self.image.set_colorkey((0, 0, 0))
            self.rect = self.image.get_rect(center=(self.x_pos, self.y_pos))
            self.mask = pygame.mask.from_surface(self.image)


    def reset_pos(self):
        if 3 < self.last_pos[0] or self.last_pos[0] < self.env.width - 2 or 3 < self.last_pos[1] or self.last_pos[1] < self.env.height - 2:
            self.x_pos = self.last_pos[0]
            self.y_pos = self.last_pos[1]
            self.theta = self.last_pos[2]

        self.draw_pos()

    def update2(self):
        if not self.state_life:
            dist = np.Inf
            RTK = (self.x_pos, self.y_pos)
            for alie in self.env.alies_RTK_group_sprite.spritedict:
                if math.sqrt((alie.y_pos - self.y_pos)**2+(alie.x_pos - self.x_pos)**2)<dist and not alie.state_life:
                    dist = math.sqrt((alie.y_pos - self.y_pos)**2+(alie.x_pos - self.x_pos)**2)
                    RTK = (alie.x_pos, alie.y_pos)
                    self.neighbor = alie.id

            ange = np.arctan2(-(RTK[1] - self.y_pos), (RTK[0] - self.x_pos))
            a = ange - self.theta
            if a > math.pi:
                a -= 2*math.pi
            if a < -math.pi:
                a += 2*math.pi
            if math.fabs(ange - self.theta) > 0.17:
                if a > 0:
                    self.theta += self.head_angle_velocity*self.dt
                else:
                    self.theta -= self.head_angle_velocity*self.dt
            if self.theta > math.pi:
                self.theta -= 2*math.pi
            if self.theta < -math.pi:
                self.theta += 2*math.pi
            self.draw_pos()
        else:
            self.image = pygame.Surface([0, 0])
            self.image.set_colorkey((0, 0, 0))
            self.rect = self.image.get_rect(center=(self.x_pos, self.y_pos))
            self.mask = pygame.mask.from_surface(self.image)

    def sesor(self):
        if not self.state_life:
            if self.env.obstacle:
                data, self.pointLidar = self.sense_obstacle()
            else:
                self.pointLidar[1:-1, 0] = np.multiply(self.pointLidarFull,
                                                       np.cos(self.angel_lidar - self.theta)) + self.x_pos
                self.pointLidar[1:-1, 1] = np.multiply(self.pointLidarFull,
                                                       np.sin(self.angel_lidar - self.theta)) + self.y_pos
                self.pointLidar[0, :] = self.x_pos, self.y_pos
                self.pointLidar[-1, :] = self.x_pos, self.y_pos
    def state(self):
        return self.x_pos, self.y_pos

    def change_start_pos(self, pos):
        self.x_pos = pos[0]
        self.y_pos = pos[1]
        self.theta = random.uniform(-math.pi, math.pi)
        self.draw_pos()

    def distance(self, obstaclePostion):
        px = (obstaclePostion[0] - self.x_pos) ** 2
        py = (obstaclePostion[1] - self.y_pos) ** 2
        return math.sqrt(px+py)


    def sense_obstacle(self):
        data = []
        points = []
        points.append((int(self.x_pos), int(self.y_pos)))
        x1, y1 = self.x_pos, self.y_pos
        for angles in self.angel_lidar:
            angle = angles + self.theta
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



    def edge_RTK(self):
        """
        angle = math.atan2(self.y_pos - self.last_pos[1], self.x_pos - self.last_pos[0])
        k = (self.y_pos - self.last_pos[1])/ (self.x_pos - self.last_pos[0])
        if -math.pi/4 < angle < math.pi/4:
            y = k * (self.last_pos[0] - 490) + self.last_pos[1]
            (self.x_pos, self.y_pos, self.theta) = (490, y, angle)
        elif math.pi/4 < angle < 3*math.pi/4:
            x = (k*self.last_pos[0] - 10 + self.last_pos[1])/k
            (self.x_pos, self.y_pos, self.theta) = (x, 10, angle)
        elif -3*math.pi/4 < angle < -math.pi/4:
            x = (k * self.last_pos[0] - 490 + self.last_pos[1]) / k
            (self.x_pos, self.y_pos, self.theta) = (x, 490, angle)
        elif 3*math.pi/4 < angle < -3*math.pi/4:
            y = k * (self.last_pos[0] - 10) + self.last_pos[1]
            (self.x_pos, self.y_pos, self.theta) = (10, y, angle)
        else:
            if angle==math.pi/4:
                (self.x_pos, self.y_pos, self.theta) = (490, 10, angle)
            elif angle==-math.pi/4:
                (self.x_pos, self.y_pos, self.theta) = (490, 490, angle)
            elif angle == -3*math.pi/4:
                (self.x_pos, self.y_pos, self.theta) = (10, 490, angle)
            else:
                (self.x_pos, self.y_pos, self.theta) = (10, 10, angle)
        """
        angle = math.atan2(self.y_pos - self.last_pos[1], self.x_pos - self.last_pos[0])
        if self.cross_edge((self.x_pos, self.y_pos), (self.last_pos[0], self.last_pos[1]), (498, 3), (498, 498))[0]:
            point = self.cross_edge((self.x_pos, self.y_pos), (self.last_pos[0], self.last_pos[1]), (498, 3),
                                    (498, 498))
            (self.x_pos, self.y_pos, self.theta) = (point[1], point[2], angle)
        elif self.cross_edge((self.x_pos, self.y_pos), (self.last_pos[0], self.last_pos[1]), (3, 3), (498, 3))[0]:
            point = self.cross_edge((self.x_pos, self.y_pos), (self.last_pos[0], self.last_pos[1]), (3, 3), (498, 3))
            (self.x_pos, self.y_pos, self.theta) = (point[1], point[2], angle)
        elif self.cross_edge((self.x_pos, self.y_pos), (self.last_pos[0], self.last_pos[1]), (3, 3), (3, 498))[0]:
            point = self.cross_edge((self.x_pos, self.y_pos), (self.last_pos[0], self.last_pos[1]), (3, 3), (3, 498))
            (self.x_pos, self.y_pos, self.theta) = (point[1], point[2], angle)
        elif self.cross_edge((self.x_pos, self.y_pos), (self.last_pos[0], self.last_pos[1]), (3, 498), (498, 498))[0]:
            point = self.cross_edge((self.x_pos, self.y_pos), (self.last_pos[0], self.last_pos[1]), (3, 498),
                                    (498, 498))
            (self.x_pos, self.y_pos, self.theta) = (point[1], point[2], angle)


        self.draw_pos()


    def draw_pos(self):
        self.rect.center = (self.x_pos, self.y_pos)
        self.image = pygame.transform.rotozoom(self.img, math.degrees(self.theta), 1)
        self.image.set_colorkey((0, 0, 0))
        self.rect = self.image.get_rect(center=(self.x_pos, self.y_pos))
        self.mask = pygame.mask.from_surface(self.image)

    def cross_edge(self, point1_1, point2_1, point1_2, point2_2):
        x1_1, y1_1 = point1_1[0], point1_1[1]
        x1_2, y1_2 = point2_1[0], point2_1[1]

        x2_1, y2_1 = point1_2[0], point1_2[1]
        x2_2, y2_2 = point2_2[0], point2_2[1]

        def point(x, y):
            if min(x2_1, x2_2) <= round(x) <= max(x2_1, x2_2) and min(y2_1, y2_2) <= round(y) <= max(y2_1, y2_2):
                print('Точка пересечения отрезков есть, координаты: ({0:f}, {1:f}).'.
                     format(x, y))
                return True
            else:
                print('Точки пересечения отрезков нет.')
                return False

        A1 = y1_1 - y1_2
        B1 = x1_2 - x1_1
        C1 = x1_1 * y1_2 - x1_2 * y1_1
        A2 = y2_1 - y2_2
        B2 = x2_2 - x2_1
        C2 = x2_1 * y2_2 - x2_2 * y2_1

        if B1 * A2 - B2 * A1 and A1:
            y = (C2 * A1 - C1 * A2) / (B1 * A2 - B2 * A1)
            x = (-C1 - B1 * y) / A1
            bol = point(x, y)
        elif B1 * A2 - B2 * A1 and A2:
            y = (C2 * A1 - C1 * A2) / (B1 * A2 - B2 * A1)
            x = (-C2 - B2 * y) / A2
            bol = point(x, y)
        else:
            print('Точки пересечения отрезков нет, отрезки ||.')
            bol = False
        return (bol, x, y)
