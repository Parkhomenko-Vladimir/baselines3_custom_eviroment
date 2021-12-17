from Enviroment import Enviroment
import pygame
import time
import random
import matplotlib.pyplot as plt
import numpy as np

obstacle_turn = True
vizualaze = True
Total_war = True
done = False
head_velocity = 0.01*10    # скорость повората башни в радианах
num_obs = 10   # количество препятствий
size_obs = [30, 40]     # размер препятствий
m_step = 500    # максимальное количество шагов
num_enemy = 3   # количество противников
num_alie = 4   # количество агентов
action = np.zeros((num_alie, 2))
Vizual = True
#for ist in range(100):
env = Enviroment(obstacle_turn, vizualaze, Total_war, head_velocity, num_obs, num_enemy, num_alie, size_obs, m_step)
state = env.reset()
    #plt.imshow(state.img)
    #plt.pause(1.5)
totalRew = 0
#for i in range(100):
while not done:
    for ist in range(num_alie):
        action[ist, :] = (random.randint(100, 175), random.uniform(-3.15, 3.15))
    state, reward, done, numstep = env.step(action)
    totalRew += reward
    print("Шаг: {2} Награда: {0}, Состояние: {1}, \n Положение РТК: \n {4} \n Положение цели: \n {5} \n Действие: \n  {3}" \
          .format(reward, done, numstep, action, state.posRobot, state.target, totalRew))

    if Vizual:
        pass


done = False
# env.draw()
#if reward == 100:
    #plt.imshow(state.img)
    #plt.pause(0.01)
plt.imshow(state.img)
plt.pause(5)
state = env.reset()
totalRew = 0
