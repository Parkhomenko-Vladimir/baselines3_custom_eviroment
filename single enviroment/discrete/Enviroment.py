import pygame
import math
import random
import os
import numpy as np
from static_obs import static_obs

from RTK import RTK_cls
from Stating import State_Env
from obstacle import Obstacle

class Enviroment():
    def __init__(self, obstacle, Viz, War, head_velocity, num_obs, num_enemy, size_obs, m_step, in_collision_rew, in_win_rew, in_defeat_rew):
        self.mode_war = War
        self.vizualaze = Viz
        self.obstacle = obstacle
        self.height = 500           # размер окна (высота)
        self.width = 500            # размер окна (высота)
        self.ever = State_Env(self.height, self.width)      # структура для данных среды (state)
        self.max_step = m_step
        self.num_enemy = num_enemy
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
        self.rew_collision = in_collision_rew
        self.rew_win = in_win_rew
        self.rew_defeat = in_defeat_rew
        # настройка pygame элементов
        if self.vizualaze:
            self.map = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Simulation Batle Robotics")
        else:
            self.map = pygame.display.set_mode((self.width, self.height), flags=pygame.HIDDEN)

        game_folder = os.path.dirname(__file__)
        img_folder = os.path.join(game_folder, 'img')
        player_img = pygame.image.load(os.path.join(img_folder, 'rtk2.png')).convert()
        self.boom = pygame.image.load(os.path.join(img_folder, 'boom.png')).convert()

        # кастомизация среды
        self.circle_radius = 15     # радиус финиша
        self.num_obstacle = num_obs       # количество препятствий
        self.size_obstacle = size_obs   # размер препятствий

        # создаем группы спрайтов
        self.obstacle_group_sprite = pygame.sprite.Group()
        self.alies_RTK_group_sprite = pygame.sprite.Group()

        self.map_obs = np.zeros((self.width, self.height))


    def step(self, action):
        self.num_step += 1
        if self.num_step > self.max_step:
            return self.ever, self.reward, True, self.num_step
        self.alies_RTK_group_sprite.update(action)

        self.map.fill(self.white)
        self.obstacle_group_sprite.draw(self.map)



        if self.mode_war:

            self.RTK.sesor()
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
            self.ever.target = np.zeros((self.num_enemy, 3))
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

        if self.vizualaze:
            pygame.display.update()
            pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.draw()
                self.done = True
                self.reward = 0

                return self.ever, self.reward, self.done, self.num_step

        state = False
        for sprite_obs in self.obstacle_group_sprite:
            G = pygame.sprite.collide_mask(self.RTK, sprite_obs)
            if G != None:
                state = True

        if self.RTK.x_pos < 3 or self.RTK.y_pos < 3 or \
           self.RTK.x_pos > self.width - 3 or self.RTK.y_pos > self.height - 3 or state:
            self.reward = self.rew_collision
            self.done = True

            return self.ever, self.reward, self.done, self.num_step


        if self.mode_war:

            if (color[0], color[1], color[2]) == (255, 0, 0):
                self.reward = self.rew_defeat
                self.done = True
                self.map.fill(self.white)
                self.obstacle_group_sprite.draw(self.map)
                #pygame.draw.polygon(self.map, (255, 0, 0, 20), self.RTK_enemy.pointLidar)

                self.RTK.draw_boom()

                for enem in self.enemy_RTK_group_sprite.spritedict:
                    enem.update2()
                    enem.sesor()
                    if len(enem.pointLidar) > 3:
                        pygame.draw.polygon(self.map, (255, 0, 0, 20), enem.pointLidar)
                self.alies_RTK_group_sprite.draw(self.map)
                self.enemy_RTK_group_sprite.draw(self.map)
                self.ever.img = pygame.surfarray.array3d(self.map)
                self.ever.img = np.transpose(self.ever.img, (1, 0, 2))

                if self.vizualaze:
                    pygame.display.update()
                    pygame.display.flip()
                return self.ever, self.reward, self.done, self.num_step
            h = 0
            con_d = np.zeros(self.num_enemy)
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
                    con_d = np.zeros(self.num_enemy)
                    for enems in self.enemy_RTK_group_sprite.spritedict:
                        con_d[enems.num] = math.sqrt(math.pow(enems.x_pos - self.RTK.x_pos, 2) + math.pow(
                            enems.y_pos - self.RTK.y_pos, 2))
                        pygame.draw.polygon(self.map, (255, 0, 0, 20), enems.pointLidar)
                    self.past_d = con_d
                    self.enemy_RTK_group_sprite.draw(self.map)
                    if len(self.enemy_RTK_group_sprite) == 0:
                        self.done = True
                    if self.vizualaze:
                        pygame.display.update()
                        pygame.display.flip()
                    self.reward = self.rew_win
                    return self.ever, self.reward, self.done, self.num_step
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
                self.reward = self.rew_win
                self.done = True

            self.past_d = con_d
            return self.ever, self.reward, self.done, self.num_step

    def reset(self):
        self.done = False
        self.num_step = 0

        game_folder = os.path.dirname(__file__)
        img_folder = os.path.join(game_folder, 'img')
        player_img = pygame.image.load(os.path.join(img_folder, 'rtk2.png')).convert()
        it = 0
        while it==0:
            it = 20

            # создаем группы спрайтов
            self.obstacle_group_sprite = pygame.sprite.Group()
            self.alies_RTK_group_sprite = pygame.sprite.Group()

            # создаем препятствия
            if self.obstacle:
                for i in range(self.num_obstacle):
                    obs = Obstacle(self.width, self.height, self.size_obstacle)
                    self.obstacle_group_sprite.add(obs)
            self.obstacle_group_sprite.add(static_obs(0, 250, [1, 500]))
            self.obstacle_group_sprite.add(static_obs(250, 0, [500, 1]))
            self.obstacle_group_sprite.add(static_obs(499, 250, [1, 500]))
            self.obstacle_group_sprite.add(static_obs(250, 499, [500, 1]))

            self.map.fill(self.white)
            self.obstacle_group_sprite.draw(self.map)
            self.map_obs= pygame.surfarray.array3d(self.map)
            self.map_obs = np.transpose(self.map_obs, (1, 0, 2))

            # создаем робота в случайной точке карте
            self.RTK = RTK_cls(self, [random.randint(50, self.width - 50), random.randint(50, self.height - 50)],
                               player_img, 40, self.head_velocity, 0, 0)
            # проверяем не попал ли робот в препятствие
            self.circle_center = (random.randint(20, self.width - 20), random.randint(20, self.height - 20))
            self.spritecircle = pygame.sprite.Sprite()
            self.spritecircle.image = pygame.Surface((35 * 2, 35 * 2))
            self.spritecircle.rect = self.spritecircle.image.get_rect()
            self.spritecircle.rect.center = self.circle_center
            SS = pygame.sprite.spritecollide(self.spritecircle, self.obstacle_group_sprite, False)

            while SS:
                # обновляем случайное положение до того пока не попадет в пустую область
                self.circle_center = (random.randint(20, self.width - 20), random.randint(20, self.height - 20))
                self.spritecircle.rect.center = self.circle_center
                SS = pygame.sprite.spritecollide(self.spritecircle, self.obstacle_group_sprite, False)

                if it==0:
                    break
                it = it - 1

            self.RTK.change_start_pos(self.circle_center)

            # добавляем робота в группу спрайтов робота
            self.alies_RTK_group_sprite.add(self.RTK)

            if self.mode_war:
                self.enemy_RTK_group_sprite = pygame.sprite.Group()
                self.past_d = np.zeros(self.num_enemy)
                for i in range(self.num_enemy):

                    self.RTK_enemy = RTK_cls(self, [random.randint(50, self.width - 50), random.randint(50, self.height - 50)],
                                             player_img, 45, self.head_velocity, 1, i)

                    Sd = np.zeros((len(self.enemy_RTK_group_sprite.spritedict)))

                    if len(Sd) == 0:
                        Sd = np.array([100])
                    self.circle_center = (random.randint(20, self.width - 20), random.randint(20, self.height - 20))
                    self.Df = math.sqrt(
                        (self.RTK.x_pos - self.circle_center[0]) ** 2 + (self.RTK.y_pos - self.circle_center[1]) ** 2)
                    for enem in self.enemy_RTK_group_sprite.spritedict:
                        Sd[enem.num] = math.sqrt(
                            (enem.x_pos - self.circle_center[0]) ** 2 + (enem.y_pos - self.circle_center[1]) ** 2)

                    self.spritecircle = pygame.sprite.Sprite()
                    self.spritecircle.image = pygame.Surface((35 * 2, 35 * 2))
                    self.spritecircle.rect = self.spritecircle.image.get_rect()
                    self.spritecircle.rect.center = self.circle_center
                    SS = pygame.sprite.spritecollide(self.spritecircle, self.obstacle_group_sprite, False)
                    while SS or self.Df < 200 or Sd.min() < 90:
                        self.circle_center = (random.randint(20, self.width - 20), random.randint(20, self.height - 20))
                        self.spritecircle.rect.center = self.circle_center
                        SS = pygame.sprite.spritecollide(self.spritecircle, self.obstacle_group_sprite, False)
                        self.Df = math.sqrt(
                            (self.RTK.x_pos - self.circle_center[0]) ** 2 + (self.RTK.y_pos - self.circle_center[1]) ** 2)
                        for enem in self.enemy_RTK_group_sprite.spritedict:
                            Sd[enem.num] = math.sqrt(
                                (enem.x_pos - self.circle_center[0]) ** 2 + (enem.y_pos - self.circle_center[1]) ** 2)
                        if it == 0:
                            break
                        it = it - 1
                    if it == 0:
                        break
                    self.RTK_enemy.change_start_pos(self.circle_center)
                    self.RTK_enemy.update2()
                    self.enemy_RTK_group_sprite.add(self.RTK_enemy)

                    self.past_d[i] = math.sqrt(math.pow(self.RTK_enemy.x_pos - self.RTK.x_pos, 2) + math.pow(
                        self.RTK_enemy.y_pos - self.RTK.y_pos, 2))

            else:
                self.circle_center = (random.randint(20, self.width-20), random.randint(20, self.height-20))

                self.spritecircle = pygame.sprite.Sprite()
                self.spritecircle.image = pygame.Surface((self.circle_radius * 2+10, self.circle_radius * 2+10))
                self.spritecircle.rect = self.spritecircle.image.get_rect()
                self.spritecircle.rect.center = self.circle_center
                SS = pygame.sprite.spritecollide(self.spritecircle, self.obstacle_group_sprite, False)
                while SS or math.sqrt(math.pow(self.circle_center[0] - self.RTK.x_pos, 2) + math.pow(self.circle_center[1] - self.RTK.y_pos, 2)) < 150:
                    self.circle_center = (random.randint(20, self.width-20), random.randint(20, self.height-20))
                    self.spritecircle.rect.center = self.circle_center
                    SS = pygame.sprite.spritecollide(self.spritecircle, self.obstacle_group_sprite, False)
                    if it == 0:
                        break
                    it = it - 1
                self.past_d = math.sqrt(
                    math.pow(self.circle_center[0] - self.RTK.x_pos, 2) + math.pow(self.circle_center[1] - self.RTK.y_pos, 2))
                self.ever.target = self.circle_center


        self.ever, reward, done, numstep = self.step(8)
        return self.ever