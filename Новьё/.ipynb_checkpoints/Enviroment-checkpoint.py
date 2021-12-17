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
    def __init__(self, obstacle, Viz, War, head_velocity, num_obs, num_enemy, num_alies, size_obs, m_step):
        self.mode_war = War
        self.vizualaze = Viz
        self.obstacle = obstacle
        self.height = 500           # размер окна (высота)
        self.width = 500            # размер окна (высота)
        self.ever = State_Env(self.height, self.width)      # структура для данных среды (state)
        self.max_step = m_step
        self.num_enemy = num_enemy
        self.num_alies = num_alies
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

        self.radius_alie = 40
        self.radius_enemy = 45

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
        self.reward = np.zeros((self.num_alies))
        self.num_step += 1
        if self.num_step > self.max_step:
            return self.ever, self.reward, True, self.num_step
        self.ever.posRobot = np.zeros((self.num_alies,4))
        for alie in self.alies_RTK_group_sprite.spritedict:
            alie.update(action[alie.id, :])
            if not alie.state_life:
                for sprite_obs in self.obstacle_group_sprite:
                    G = pygame.sprite.collide_mask(alie, sprite_obs)
                    if G != None:
                        self.reward[alie.id] += -30
                        alie.reset_pos()
                for alie2 in self.alies_RTK_group_sprite.spritedict:
                    if alie2.id != alie.id:
                        G = pygame.sprite.collide_mask(alie, alie2)
                        if G != None:
                            self.reward[alie.id] += -30
                            alie.reset_pos()

            self.ever.posRobot[alie.id, :] = np.array((alie.x_pos, alie.y_pos, alie.theta, alie.state_life))

        self.map.fill(self.white)
        self.obstacle_group_sprite.draw(self.map)


        if self.mode_war:

            for alie in self.alies_RTK_group_sprite.spritedict:
                alie.sesor()
            self.map.fill(self.white)
            self.obstacle_group_sprite.draw(self.map)
            for enem in self.enemy_RTK_group_sprite.spritedict:
                enem.update2()
                enem.sesor()
                if len(enem.pointLidar) > 3 and not enem.state_life:
                    pygame.draw.polygon(self.map, (255, 0, 0, 20), enem.pointLidar)
            color = np.empty([len(self.alies_RTK_group_sprite.spritedict), 4])
            for alie in self.alies_RTK_group_sprite.spritedict:
                if not alie.state_life:
                    color[alie.id][:] = self.map.get_at((int(alie.x_pos), int(alie.y_pos)))

            self.map.fill(self.white)
            self.obstacle_group_sprite.draw(self.map)




            h = 0
            color1 = np.empty([len(self.enemy_RTK_group_sprite.spritedict), 4])
            for alie in self.alies_RTK_group_sprite.spritedict:
                if len(alie.pointLidar) > 3 and not alie.state_life:
                    pygame.draw.polygon(self.map, (0, 0, 255, 20), alie.pointLidar)
                    for enem in self.enemy_RTK_group_sprite.spritedict:
                        if not enem.state_life:
                            color1[enem.num][:] = self.map.get_at((int(enem.x_pos), int(enem.y_pos)))
                            if (color1[enem.num][0], color1[enem.num][1], color1[enem.num][2]) == (
                            0, 0, 255) and not enem.state_life:
                                self.reward[alie.id] += 100

                h += 1
            self.ever.target = np.zeros((self.num_enemy, 4))
            for enem in self.enemy_RTK_group_sprite.spritedict:
                if len(enem.pointLidar) > 3 and not enem.state_life:
                    pygame.draw.polygon(self.map, (255, 0, 0, 20), enem.pointLidar)
                self.ever.target[enem.num, :] = np.array((enem.x_pos, enem.y_pos, enem.theta, enem.state_life))

        else:
            self.alies_RTK_group_sprite.draw(self.map)
            pygame.draw.circle(self.map, self.green, self.circle_center, self.circle_radius)

            self.ever.target = np.array(self.circle_center)

        self.ever.img = pygame.surfarray.array3d(self.map)
        self.ever.img = np.transpose(self.ever.img, (1, 0, 2))

        #self.ever.posRobot = np.array((self.RTK.x_pos, self.RTK.y_pos, self.RTK.theta))



        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.draw()
                self.done = True
                self.reward = 0

                return self.ever, self.reward, self.done, self.num_step
        if self.mode_war:
            for alie in self.alies_RTK_group_sprite.spritedict:
                if (color[alie.id][0], color[alie.id][1], color[alie.id][2]) == (255, 0, 0) and not alie.state_life:
                    alie.state_life = True
                    self.map.fill(self.white)
                    self.obstacle_group_sprite.draw(self.map)
                    for enem in self.enemy_RTK_group_sprite.spritedict:
                        if not enem.state_life:
                            pygame.draw.polygon(self.map, (255, 0, 0, 20), enem.pointLidar)
                    #self.enemy_RTK_group_sprite.draw(self.map)
                    alie.draw_boom()
                    self.map.blit(alie.image, alie.rect)
                    self.ever.img = pygame.surfarray.array3d(self.map)
                    self.ever.img = np.transpose(self.ever.img, (1, 0, 2))
                    #self.alies_RTK_group_sprite.remove(alie)

                    for alien in self.alies_RTK_group_sprite.spritedict:
                        if not alien.state_life:
                            pygame.draw.polygon(self.map, (0, 0, 255, 20), alien.pointLidar)

                    self.reward[alie.id] = -100
                    self.alies_RTK_group_sprite.draw(self.map)
                    self.ever.posRobot[alie.id, :] = np.NAN
                    self.ever.posRobot[alie.id, 3] = True
                    [alie.x_pos, alie.y_pos, alie.theta] = [np.NAN, np.NAN, np.NAN]
                        #return self.ever, self.reward, self.done, self.num_step

                    #alie.kill()



            h = 0
            con_d = np.zeros(self.num_enemy)
            # self.reward = 0
            for enem in self.enemy_RTK_group_sprite.spritedict:
                con_d[enem.num] = math.sqrt(math.pow(enem.x_pos - self.RTK.x_pos, 2) + math.pow(
                    enem.y_pos - self.RTK.y_pos, 2))

                if (color1[enem.num][0], color1[enem.num][1], color1[enem.num][2]) == (0, 0, 255) and not enem.state_life:

                    enem.state_life = True
                    self.map.fill(self.white)
                    self.obstacle_group_sprite.draw(self.map)
                    self.alies_RTK_group_sprite.draw(self.map)
                    enem.draw_boom()
                    self.map.blit(enem.image, enem.rect)
                    for alie in self.alies_RTK_group_sprite.spritedict:
                        if not alie.state_life:
                            pygame.draw.polygon(self.map, (0, 0, 255, 20), alie.pointLidar)
                    #self.enemy_RTK_group_sprite.remove(enem)

                    con_d = np.zeros(self.num_enemy)
                    for enems in self.enemy_RTK_group_sprite.spritedict:
                        if not enems.state_life:
                            pygame.draw.polygon(self.map, (255, 0, 0, 20), enems.pointLidar)
                    self.ever.img = pygame.surfarray.array3d(self.map)
                    self.ever.img = np.transpose(self.ever.img, (1, 0, 2))
                    self.past_d = con_d
                    self.enemy_RTK_group_sprite.draw(self.map)
                    #self.reward[enem.neighbor] = 100
                    self.ever.target[enem.num, :] = np.NAN
                    self.ever.target[enem.num, 3] = True
                    [enem.x_pos, enem.y_pos, enem.theta] = [np.NAN, np.NAN, np.NAN]

            self.enemy_RTK_group_sprite.draw(self.map)
            self.alies_RTK_group_sprite.draw(self.map)
            if self.vizualaze:
                pygame.display.update()
                pygame.display.flip()
            if all(self.ever.posRobot[:, 3]) or all(self.ever.target[:, 3]):
                self.done = True
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
            self.map_obs = pygame.surfarray.array3d(self.map)
            self.map_obs = np.transpose(self.map_obs, (1, 0, 2))
            for i in range(self.num_alies):
                # создаем робота в случайной точке карте

                # проверяем не попал ли робот в препятствие
                self.circle_center = (random.randint(20, self.width - 20), random.randint(20, self.height - 20))
                self.RTK = RTK_cls(self, self.circle_center,
                                   player_img, self.radius_alie, self.head_velocity, i)
                self.spritecircle = pygame.sprite.Sprite()
                self.spritecircle.image = pygame.Surface((35 * 2, 35 * 2))
                self.spritecircle.rect = self.spritecircle.image.get_rect()
                self.spritecircle.rect.center = self.circle_center
                SS = pygame.sprite.spritecollide(self.spritecircle, self.obstacle_group_sprite, False)
                self.RTK.id = i
                if i:
                    Df = np.zeros(len(self.alies_RTK_group_sprite.spritedict))
                else:
                    Df = np.array([self.radius_alie])
                while SS or Df.min() < self.radius_alie:
                    # обновляем случайное положение до того пока не попадет в пустую область
                    self.circle_center = (random.randint(20, self.width - 20), random.randint(20, self.height - 20))
                    self.spritecircle.rect.center = self.circle_center
                    SS = pygame.sprite.spritecollide(self.spritecircle, self.obstacle_group_sprite, False)
                    for alie in self.alies_RTK_group_sprite.spritedict:
                        Df[alie.id] = math.sqrt((alie.x_pos-self.circle_center[0])**2 + (alie.y_pos-self.circle_center[1])**2)
                    if it == 0:
                        break
                    it = it - 1
                if it == 0:
                    self.alies_RTK_group_sprite.empty()
                    break
                self.RTK.change_start_pos(self.circle_center)
                # добавляем робота в группу спрайтов робота
                self.alies_RTK_group_sprite.add(self.RTK)

            if self.mode_war:
                self.enemy_RTK_group_sprite = pygame.sprite.Group()
                self.past_d = np.zeros(self.num_enemy)
                for i in range(self.num_enemy):
                    self.spritecircle = pygame.sprite.Sprite()
                    self.spritecircle.image = pygame.Surface((40 * 2, 40 * 2))
                    self.spritecircle.rect = self.spritecircle.image.get_rect()
                    self.circle_center = (random.randint(20, self.width - 20), random.randint(20, self.height - 20))
                    self.spritecircle.rect.center = self.circle_center
                    self.RTK_enemy = RTK_cls(self, self.circle_center, player_img, self.radius_enemy, self.head_velocity, i)

                    Sd = np.zeros((len(self.enemy_RTK_group_sprite.spritedict)))

                    if len(Sd) == 0:
                        Sd = np.array([100])
                    Df = np.zeros((self.num_alies))
                    for RTK in self.alies_RTK_group_sprite.spritedict:
                        Df[RTK.id] = math.sqrt(
                            (RTK.x_pos - self.circle_center[0]) ** 2 + (RTK.y_pos - self.circle_center[1]) ** 2)
                    for enem in self.enemy_RTK_group_sprite.spritedict:
                        Sd[enem.num] = math.sqrt(
                            (enem.x_pos - self.circle_center[0]) ** 2 + (enem.y_pos - self.circle_center[1]) ** 2)


                    SS = pygame.sprite.spritecollide(self.spritecircle, self.obstacle_group_sprite, False)
                    while SS or Df.min() < 2*self.radius_enemy or Sd.min() < self.radius_enemy*2:
                        self.circle_center = (random.randint(20, self.width - 20), random.randint(20, self.height - 20))
                        self.spritecircle.rect.center = self.circle_center
                        SS = pygame.sprite.spritecollide(self.spritecircle, self.obstacle_group_sprite, False)
                        for RTK in self.alies_RTK_group_sprite.spritedict:
                            Df[RTK.id] = math.sqrt(
                                (RTK.x_pos - self.circle_center[0]) ** 2 + (RTK.y_pos - self.circle_center[1]) ** 2)
                        for enem in self.enemy_RTK_group_sprite.spritedict:
                            Sd[enem.num] = math.sqrt(
                                (enem.x_pos - self.circle_center[0]) ** 2 + (enem.y_pos - self.circle_center[1]) ** 2)
                        if it == 0:
                            break

                        it = it - 1
                    if it == 0:
                        self.enemy_RTK_group_sprite.empty()
                        break
                    self.RTK_enemy.change_start_pos(self.circle_center)
                    self.RTK_enemy.update2()
                    self.enemy_RTK_group_sprite.add(self.RTK_enemy)

                    #self.past_d[i] = math.sqrt(math.pow(self.RTK_enemy.x_pos - self.RTK.x_pos, 2) + math.pow(
                       # self.RTK_enemy.y_pos - self.RTK.y_pos, 2))

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

        action = np.full((self.num_alies, 2), 0)
        self.ever, reward, done, numstep = self.step(action)
        return self.ever