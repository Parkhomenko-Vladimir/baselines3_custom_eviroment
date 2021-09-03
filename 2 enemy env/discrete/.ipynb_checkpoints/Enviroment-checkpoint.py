import pygame
import math
import random
import os
import gym
import time
import numpy as np
from gym import Env, spaces
from RTK import RTK_cls
from Stating import State_Env
from obstacle import Obstacle

class Enviroment(Env):
    def __init__(self, obstacle, Viz, War):
        self.mode_war = War
        self.vizualaze = Viz
        self.obstacle = obstacle
        self.height = 500           # размер окна (высота)
        self.width = 500            # размер окна (высота)
        self.ever = State_Env(self.height, self.width)      # структура для данных среды (state)
        # цвета
        self.black = (0, 0, 0)
        self.white = (255, 255, 255)
        self.green = (0, 255, 0)
        self.blue = (0, 0, 255)
        self.red = (255, 0, 0)
        self.yel = (255, 255, 0)
        # настройка выходных данных
        self.done = True
        self.reward = 0
        self.koef = 3
        self.num_step = 0
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
        self.num_obstacle = 7       # количество препятствий
        self.size_obstacle = [40, 70]   # размер препятствий

        # создаем группы спрайтов
        self.obstacle_group_sprite = pygame.sprite.Group()
        self.alies_RTK_group_sprite = pygame.sprite.GroupSingle()

        # создаем препятствия
        if self.obstacle:
            for i in range(self.num_obstacle):
                obs = Obstacle(self.width, self.height, self.size_obstacle)
                self.obstacle_group_sprite.add(obs)


    def step(self, action):
        self.num_step += 1
        if self.num_step > 1500:
            return self.ever, self.reward, False, self.num_step
        self.alies_RTK_group_sprite.update(action)

        self.map.fill(self.white)
        self.obstacle_group_sprite.draw(self.map)

        self.RTK.sesor()

        if self.mode_war:
            self.RTK_enemy.update2()
            self.RTK_enemy.sesor()
            self.map.fill(self.white)
            self.obstacle_group_sprite.draw(self.map)

            if len(self.RTK_enemy.pointLidar) > 3:
                pygame.draw.polygon(self.map, (255, 0, 0, 20), self.RTK_enemy.pointLidar)
            if self.RTK.x_pos > 2 and self.RTK.y_pos > 2 and self.RTK.x_pos < self.width -2  and self.RTK.y_pos < self.height - 2:
                color = self.map.get_at((int(self.RTK.x_pos), int(self.RTK.y_pos)))
            self.map.fill(self.white)
            self.obstacle_group_sprite.draw(self.map)
            if len(self.RTK.pointLidar) > 3:
                pygame.draw.polygon(self.map, (0, 0, 255, 20), self.RTK.pointLidar)
            if self.RTK_enemy.x_pos > 2 and self.RTK_enemy.y_pos > 2 and \
                    self.RTK.x_pos < self.width - 2 and self.RTK.y_pos < self.height - 2:
                color1 = self.map.get_at((int(self.RTK_enemy.x_pos), int(self.RTK_enemy.y_pos)))
            if len(self.RTK_enemy.pointLidar) > 3:
                pygame.draw.polygon(self.map, (255, 0, 0, 20), self.RTK_enemy.pointLidar)

            self.alies_RTK_group_sprite.draw(self.map)
            self.enemy_RTK_group_sprite.draw(self.map)



            self.ever.target = (self.RTK_enemy.x_pos, self.RTK_enemy.y_pos, self.RTK_enemy.theta)

        else:
            self.alies_RTK_group_sprite.draw(self.map)
            pygame.draw.circle(self.map, self.green, self.circle_center, self.circle_radius)

            self.ever.target = self.circle_center

        self.ever.img = pygame.surfarray.array3d(self.map)
        self.ever.img = np.transpose(self.ever.img, (1, 0, 2))
        self.ever.posRobot = [self.RTK.x_pos, self.RTK.y_pos, self.RTK.theta]

        S = pygame.sprite.spritecollide(self.RTK, self.obstacle_group_sprite, False)
        if self.vizualaze:
            pygame.display.update()
            pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.draw()
                self.done = False
                self.reward = 0

                return self.ever, self.reward, self.done, self.num_step

        if self.RTK.x_pos < 3 or self.RTK.y_pos < 3 or \
           self.RTK.x_pos > self.width - 3 or self.RTK.y_pos > self.height - 3 or S:
            self.reward = -30.0
            self.done = False

            return self.ever, self.reward, self.done, self.num_step



        if self.mode_war:
            con_d = math.sqrt(math.pow(self.RTK_enemy.x_pos - self.RTK.x_pos, 2) + math.pow(
                self.RTK_enemy.y_pos - self.RTK.y_pos, 2))
            if (color[0], color[1], color[2]) == (255, 0, 0):
                self.reward = -100
                self.done = False
                return self.ever, self.reward, self.done, self.num_step
            if (color1[0], color1[1], color1[2]) == (0, 0, 255):
                self.reward = 100
                self.done = False
                self.map.fill(self.white)
                self.obstacle_group_sprite.draw(self.map)
                pygame.draw.polygon(self.map, (0, 0, 255, 20), self.RTK.pointLidar)
                self.alies_RTK_group_sprite.draw(self.map)
                self.RTK_enemy.draw_boom()
                self.enemy_RTK_group_sprite.draw(self.map)
                self.ever.img = pygame.surfarray.array3d(self.map)
                self.ever.img = np.transpose(self.ever.img, (1, 0, 2))
                return self.ever, self.reward, self.done, self.num_step
            self.reward = -0.1 + self.koef * (self.past_d - con_d)
            self.past_d = con_d
            return self.ever, self.reward, self.done, self.num_step

        else:

            con_d = math.sqrt(math.pow(self.circle_center[0] - self.RTK.x_pos, 2) + math.pow(
                self.circle_center[1] - self.RTK.y_pos, 2))
            S = pygame.sprite.collide_mask(self.RTK, self.spritecircle)
            if S:
                self.reward = 100
                self.done = False
            self.reward = -0.1 + self.koef * (self.past_d - con_d)
            self.past_d = con_d
            return self.ever, self.reward, self.done, self.num_step

    def draw(self):
        self.map.fill(self.white)
        self.obstacle_group_sprite.draw(self.map)
        if self.mode_war:
            if len(self.RTK.pointLidar) > 3:
                pygame.draw.polygon(self.map, (0, 0, 255, 20), self.RTK.pointLidar)
            if len(self.RTK_enemy.pointLidar) > 3:
                pygame.draw.polygon(self.map, (255, 0, 0, 20), self.RTK_enemy.pointLidar)

            self.alies_RTK_group_sprite.draw(self.map)
            self.enemy_RTK_group_sprite.draw(self.map)
        else:
            pygame.draw.circle(self.map, self.green, self.circle_center, self.circle_radius)
        self.alies_RTK_group_sprite.draw(self.map)
        pygame.display.update()
        pygame.display.flip()
        time.sleep(1)

    def reset(self):
        self.done = True
        self.num_step = 0
        game_folder = os.path.dirname(__file__)
        img_folder = os.path.join(game_folder, 'img')
        player_img = pygame.image.load(os.path.join(img_folder, 'rtk.png')).convert()

        # кастомизация среды
        self.circle_radius = 10  # радиус финиша
        self.num_obstacle = 7  # количество препятствий
        self.size_obstacle = [40, 70]  # размер препятствий

        # создаем группы спрайтов
        self.obstacle_group_sprite = pygame.sprite.Group()
        self.alies_RTK_group_sprite = pygame.sprite.GroupSingle()

        # создаем препятствия
        if self.obstacle:
            for i in range(self.num_obstacle):
                obs = Obstacle(self.width, self.height, self.size_obstacle)
                self.obstacle_group_sprite.add(obs)

        # создаем робота в случайной точке карте
        self.RTK = RTK_cls(self, [random.randint(50, self.width - 50), random.randint(50, self.height - 50)],
                           player_img)
        # проверяем не попал ли робот в препятствие
        SS = pygame.sprite.spritecollide(self.RTK, self.obstacle_group_sprite, False)
        while SS:
            # обновляем случайное положение до того пока не попадет в пустую область
            self.RTK.change_start_pos([random.randint(50, self.width - 50), random.randint(50, self.height - 50)])
            SS = pygame.sprite.spritecollide(self.RTK, self.obstacle_group_sprite, False)
        # добавляем робота в группу спрайтов робота
        self.alies_RTK_group_sprite.add(self.RTK)
        if self.mode_war:
            self.enemy_RTK_group_sprite = pygame.sprite.GroupSingle()
            self.RTK_enemy = RTK_cls(self, [random.randint(50, self.width - 50), random.randint(50, self.height - 50)],
                                     player_img)
            SS = pygame.sprite.spritecollide(self.RTK_enemy, self.obstacle_group_sprite, False)
            self.Df = math.sqrt(
                (self.RTK.x_pos - self.RTK_enemy.x_pos) ** 2 + (self.RTK.y_pos - self.RTK_enemy.y_pos) ** 2)
            while SS or math.sqrt(
                    (self.RTK.x_pos - self.RTK_enemy.x_pos) ** 2 + (self.RTK.y_pos - self.RTK_enemy.y_pos) ** 2) < 200:
                self.RTK_enemy.change_start_pos(
                    [random.randint(50, self.width - 50), random.randint(50, self.height - 50)])
                SS = pygame.sprite.spritecollide(self.RTK_enemy, self.obstacle_group_sprite, False)
                self.Df = math.sqrt(
                    (self.RTK.x_pos - self.RTK_enemy.x_pos) ** 2 + (self.RTK.y_pos - self.RTK_enemy.y_pos) ** 2)

            self.enemy_RTK_group_sprite.add(self.RTK_enemy)
            self.enemy_RTK_group_sprite.update(random.randint(0, 7))
            self.past_d = math.sqrt(math.pow(self.RTK_enemy.x_pos - self.RTK.x_pos, 2) + math.pow(
                self.RTK_enemy.y_pos - self.RTK.y_pos, 2))
            self.ever.target = (self.RTK_enemy.x_pos, self.RTK_enemy.y_pos)
        else:
            self.circle_center = (random.randint(0, self.width), random.randint(0, self.height))
            self.circle_radius = 10

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


        self.ever.img = pygame.surfarray.array3d(self.map)
        self.ever.img = np.transpose(self.ever.img, (1, 0, 2))
        self.ever.posRobot = [self.RTK.x_pos, self.RTK.y_pos]
        self.ever, reward, done, numstep = self.step(random.randint(0, 7))
        return self.ever