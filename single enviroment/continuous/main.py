from Enviroment import Enviroment
import pygame
import time
import random
import matplotlib.pyplot as plt

obstacle_turn = True
vizualaze = True
Total_war = True
done = False
head_velocity = 0.01    # скорость повората башни в радианах
num_obs = 10   # количество препятствий
size_obs = [10, 20]     # размер препятствий
m_step = 500    # максимальное количество шагов
num_enemy = 1   # количество противников
rew_col = -40
rew_win = 23600
rew_defeat = -600


env = Enviroment(obstacle_turn, vizualaze, Total_war, head_velocity, num_obs, num_enemy, size_obs, m_step, rew_col, rew_win, rew_defeat)
state = env.reset()
    #plt.imshow(state.img)
    #plt.pause(1.5)
totalRew = 0
for i in range(100):
    while not done:

        action = (random.randint(10, 50), random.uniform(-1.5, 1.5))
        state, reward, done, numstep = env.step(action)
        totalRew += reward
        print("Шаг: {2} Награда: {0}, Состояние: {1}, Положение РТК: {4} Положение цели: {5} Действие: {3}" \
              .format(reward, done, numstep, action, state.posRobot, state.target, totalRew))



    done = False
    # env.draw()
    if reward == 100:
        #plt.imshow(state.img)
        plt.pause(0.01)
    state = env.reset()
    totalRew = 0
