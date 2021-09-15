import pygame
import math
import random
import os
import gym
import time
import numpy as np

from RTK import RTK_cls
from Stating import State_Env
from obstacle import Obstacle

class Enviroment():
    def __init__(self, obstacle, Viz, War, head_velocity, num_obs, size_obs, m_step):
        self.mode_war = War
        self.vizualaze = Viz
        self.obstacle = obstacle
        self.height = 500           # размер окна (высота)
        self.width = 500            # размер окна (высота)
        self.ever = State_Env(self.height, self.width)      # структура для данных среды (state)
        self.max_step = m_step
        # цвета
        self.black = (0, 0, 0)
        self.white = (255, 255, 255)
        self.green = (0, 255, 0)
        self.blue = (0, 0, 255)
        self.red = (255, 0, 0)
        self.yel = (255, 255, 0)
        # настройка выходных данных
        self.done = False
        self.reward = 0
        self.koef = 3
        self.num_step = 0
        self.head_velocity = head_velocity
        # настройка pygame элементов
        if self.vizualaze:
            self.map = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Simulation Batle Robotics")
        else:
            self.map = pygame.display.set_mode((self.width, self.height), flags=pygame.HIDDEN)

        game_folder = os.path.dirname(__file__)
        img_folder = os.path.join(game_folder, 'img')
        player_img = pygame.image.load(os.path.join(img_folder, 'rtk.png')).convert()
        self.boom = pygame.image.load(os.path.join(img_folder, 'boom.png')).convert()

        # кастомизация среды
        self.circle_radius = 10     # радиус финиша
        self.num_obstacle = num_obs       # количество препятствий
        self.size_obstacle = size_obs   # размер препятствий

        # создаем группы спрайтов
        self.obstacle_group_sprite = pygame.sprite.Group()
        self.alies_RTK_group_sprite = pygame.sprite.Group()

        # создаем препятствия
        if self.obstacle:
            for i in range(self.num_obstacle):
                obs = Obstacle(self.width, self.height, self.size_obstacle)
                self.obstacle_group_sprite.add(obs)


    def step(self, action):
        self.num_step += 1
        if self.num_step > self.max_step:
            return self.ever, self.reward, True, self.num_step
        self.alies_RTK_group_sprite.update(action)

        self.map.fill(self.white)
        self.obstacle_group_sprite.draw(self.map)

        self.RTK.sesor()

        if self.mode_war:


            self.map.fill(self.white)
            self.obstacle_group_sprite.draw(self.map)
            for enem in self.enemy_RTK_group_sprite.spritedict:
                enem.update2()
                enem.sesor()
                if len(enem.pointLidar) > 3:
                    pygame.draw.polygon(self.map, (255, 0, 0, 20), enem.pointLidar)
            if self.RTK.x_pos > 2 and self.RTK.y_pos > 2 and self.RTK.x_pos < self.width -2  and self.RTK.y_pos < self.height - 2:
                color = self.map.get_at((int(self.RTK.x_pos), int(self.RTK.y_pos)))
            self.map.fill(self.white)
            self.obstacle_group_sprite.draw(self.map)
            if len(self.RTK.pointLidar) > 3:
                pygame.draw.polygon(self.map, (0, 0, 255, 20), self.RTK.pointLidar)
            h = 0
            color1 = np.empty([len(self.enemy_RTK_group_sprite.spritedict), 4])

            for enem in self.enemy_RTK_group_sprite.spritedict:
                if enem.x_pos > 2 and enem.y_pos > 2 and \
                        enem.x_pos < self.width - 2 and enem.y_pos < self.height - 2:
                    color1[h][:] = self.map.get_at((int(enem.x_pos), int(enem.y_pos)))

                h += 1
            self.ever.target = np.zeros((2, 3))
            for enem in self.enemy_RTK_group_sprite.spritedict:
                if len(enem.pointLidar) > 3:
                    pygame.draw.polygon(self.map, (255, 0, 0, 20), enem.pointLidar)
                self.ever.target[enem.num, :] = np.array((enem.x_pos, enem.y_pos, enem.theta))


            self.alies_RTK_group_sprite.draw(self.map)
            self.enemy_RTK_group_sprite.draw(self.map)


        else:
            self.alies_RTK_group_sprite.draw(self.map)
            pygame.draw.circle(self.map, self.green, self.circle_center, self.circle_radius)

            self.ever.target = np.array(self.circle_center)

        self.ever.img = pygame.surfarray.array3d(self.map)
        self.ever.img = np.transpose(self.ever.img, (1, 0, 2))
        self.ever.posRobot = np.array((self.RTK.x_pos, self.RTK.y_pos, self.RTK.theta))

        S = pygame.sprite.spritecollide(self.RTK, self.obstacle_group_sprite, False)
        if self.vizualaze:
            pygame.display.update()
            pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.draw()
                self.done = True
                self.reward = 0

                return self.ever, self.reward, self.done, self.num_step

        if self.RTK.x_pos < 3 or self.RTK.y_pos < 3 or \
           self.RTK.x_pos > self.width - 3 or self.RTK.y_pos > self.height - 3 or S:
            self.reward = -30.0
            self.done = True

            return self.ever, self.reward, self.done, self.num_step



        if self.mode_war:

            if (color[0], color[1], color[2]) == (255, 0, 0):
                self.reward = -100
                self.done = True
                self.map.fill(self.white)
                self.obstacle_group_sprite.draw(self.map)
                #pygame.draw.polygon(self.map, (255, 0, 0, 20), self.RTK_enemy.pointLidar)
                self.enemy_RTK_group_sprite.draw(self.map)
                self.RTK.draw_boom()
                self.alies_RTK_group_sprite.draw(self.map)
                self.ever.img = pygame.surfarray.array3d(self.map)
                self.ever.img = np.transpose(self.ever.img, (1, 0, 2))
                if self.vizualaze:
                    pygame.display.update()
                    pygame.display.flip()
                return self.ever, self.reward, self.done, self.num_step
            h = 0
            con_d = [0, 0]
            self.reward = 0
            for enem in self.enemy_RTK_group_sprite.spritedict:
                con_d[enem.num] = math.sqrt(math.pow(enem.x_pos - self.RTK.x_pos, 2) + math.pow(
                    enem.y_pos - self.RTK.y_pos, 2))

                if (color1[h][0], color1[h][1], color1[h][2]) == (0, 0, 255):

                    enem.state_life = False
                    self.map.fill(self.white)
                    self.obstacle_group_sprite.draw(self.map)
                    pygame.draw.polygon(self.map, (0, 0, 255, 20), self.RTK.pointLidar)
                    self.alies_RTK_group_sprite.draw(self.map)
                    enem.draw_boom()
                    self.map.blit(enem.image, enem.rect)
                    self.ever.img = pygame.surfarray.array3d(self.map)
                    self.ever.img = np.transpose(self.ever.img, (1, 0, 2))
                    self.enemy_RTK_group_sprite.remove(enem)
                    con_d = [0, 0]
                    for enem in self.enemy_RTK_group_sprite.spritedict:
                        con_d[enem.num] = math.sqrt(math.pow(enem.x_pos - self.RTK.x_pos, 2) + math.pow(
                            enem.y_pos - self.RTK.y_pos, 2))
                        pygame.draw.polygon(self.map, (255, 0, 0, 20), enem.pointLidar)
                    self.past_d = con_d
                    self.enemy_RTK_group_sprite.draw(self.map)
                    if len(self.enemy_RTK_group_sprite) == 0:
                        self.done = True
                    if self.vizualaze:
                        pygame.display.update()
                        pygame.display.flip()
                    return self.ever, 100, self.done, self.num_step
                self.reward += -0.1 + self.koef * (self.past_d[enem.num] - con_d[enem.num])
                h += 1

            self.past_d = con_d
            return self.ever, self.reward, self.done, self.num_step

        else:

            con_d = math.sqrt(math.pow(self.circle_center[0] - self.RTK.x_pos, 2) + math.pow(
                self.circle_center[1] - self.RTK.y_pos, 2))
            S = pygame.sprite.collide_mask(self.RTK, self.spritecircle)
            self.reward = -0.1 + self.koef * (self.past_d - con_d)
            if S:
                self.reward = 100
                self.done = True

            self.past_d = con_d
            return self.ever, self.reward, self.done, self.num_step

    def draw(self):
        self.map.fill(self.white)
        self.obstacle_group_sprite.draw(self.map)
        if self.mode_war:
            # if len(self.RTK.pointLidar) > 3:
            #     pygame.draw.polygon(self.map, (0, 0, 255, 20), self.RTK.pointLidar)
            # if len(self.RTK_enemy.pointLidar) > 3:
            #     pygame.draw.polygon(self.map, (255, 0, 0, 20), self.RTK_enemy.pointLidar)

            self.alies_RTK_group_sprite.draw(self.map)
            self.enemy_RTK_group_sprite.draw(self.map)
        else:
            pygame.draw.circle(self.map, self.green, self.circle_center, self.circle_radius)
        self.alies_RTK_group_sprite.draw(self.map)
        pygame.display.update()
        pygame.display.flip()
        time.sleep(1)

    def reset(self):
        self.done = False
        self.num_step = 0
        game_folder = os.path.dirname(__file__)
        img_folder = os.path.join(game_folder, 'img')
        player_img = pygame.image.load(os.path.join(img_folder, 'rtk.png')).convert()

        # кастомизация среды
        self.circle_radius = 10  # радиус финиша
        # self.num_obstacle = 7  # количество препятствий
        # self.size_obstacle = [40, 70]  # размер препятствий

        # создаем группы спрайтов
        self.obstacle_group_sprite = pygame.sprite.Group()
        self.alies_RTK_group_sprite = pygame.sprite.Group()

        # создаем препятствия
        if self.obstacle:
            for i in range(self.num_obstacle):
                obs = Obstacle(self.width, self.height, self.size_obstacle)
                self.obstacle_group_sprite.add(obs)

        # создаем робота в случайной точке карте
        self.RTK = RTK_cls(self, [random.randint(50, self.width - 50), random.randint(50, self.height - 50)],
                           player_img, 90, self.head_velocity, 0, 0)
        # проверяем не попал ли робот в препятствие
        SS = pygame.sprite.spritecollide(self.RTK, self.obstacle_group_sprite, False)
        while SS:
            # обновляем случайное положение до того пока не попадет в пустую область
            self.RTK.change_start_pos([random.randint(50, self.width - 50), random.randint(50, self.height - 50)])
            SS = pygame.sprite.spritecollide(self.RTK, self.obstacle_group_sprite, False)
        # добавляем робота в группу спрайтов робота
        self.alies_RTK_group_sprite.add(self.RTK)
        if self.mode_war:
            self.enemy_RTK_group_sprite = pygame.sprite.Group()
            self.past_d = [0, 0]
            for i in range(2):

                self.RTK_enemy = RTK_cls(self, [random.randint(50, self.width - 50), random.randint(50, self.height - 50)],
                                         player_img, 100, self.head_velocity, 1, i)
                SS = pygame.sprite.spritecollide(self.RTK_enemy, self.obstacle_group_sprite, False)
                Sd = np.zeros((len(self.enemy_RTK_group_sprite.spritedict)))
                for enem in self.enemy_RTK_group_sprite.spritedict:
                    Sd[enem.num] = math.sqrt(
                        (enem.x_pos - self.RTK_enemy.x_pos) ** 2 + (enem.y_pos - self.RTK_enemy.y_pos) ** 2)
                self.Df = math.sqrt(
                    (self.RTK.x_pos - self.RTK_enemy.x_pos) ** 2 + (self.RTK.y_pos - self.RTK_enemy.y_pos) ** 2)
                if len(Sd) == 0:
                    Sd = np.array([100])

                while SS or self.Df < 200 or Sd.min() < 60:
                    self.RTK_enemy.change_start_pos(
                        [random.randint(50, self.width - 50), random.randint(50, self.height - 50)])
                    SS = pygame.sprite.spritecollide(self.RTK_enemy, self.obstacle_group_sprite, False)
                    self.Df = math.sqrt(
                        (self.RTK.x_pos - self.RTK_enemy.x_pos) ** 2 + (self.RTK.y_pos - self.RTK_enemy.y_pos) ** 2)
                    for enem in self.enemy_RTK_group_sprite.spritedict:
                        Sd[enem.num] = math.sqrt(
                            (enem.x_pos - self.RTK_enemy.x_pos) ** 2 + (enem.y_pos - self.RTK_enemy.y_pos) ** 2)
                self.RTK_enemy.update(random.randint(0, 7))
                self.enemy_RTK_group_sprite.add(self.RTK_enemy)

                self.past_d[i] = math.sqrt(math.pow(self.RTK_enemy.x_pos - self.RTK.x_pos, 2) + math.pow(
                    self.RTK_enemy.y_pos - self.RTK.y_pos, 2))

        else:
            self.circle_center = (random.randint(0, self.width), random.randint(0, self.height))
            self.circle_radius = 50

            self.spritecircle = pygame.sprite.Sprite()
            self.spritecircle.image = pygame.Surface((self.circle_radius * 2, self.circle_radius * 2))
            self.spritecircle.rect = self.spritecircle.image.get_rect()
            self.spritecircle.rect.center = self.circle_center
            SS = pygame.sprite.spritecollide(self.spritecircle, self.obstacle_group_sprite, False)
            while SS:
                self.circle_center = (random.randint(0, self.width), random.randint(0, self.height))
                self.spritecircle.rect.center = self.circle_center
                SS = pygame.sprite.spritecollide(self.spritecircle, self.obstacle_group_sprite, False)
            self.past_d = math.sqrt(
                math.pow(self.circle_center[0] - self.RTK.x_pos, 2) + math.pow(self.circle_center[1] - self.RTK.y_pos, 2))
            self.ever.target = self.circle_center


        self.ever, reward, done, numstep = self.step(random.randint(0, 7))
        return self.ever