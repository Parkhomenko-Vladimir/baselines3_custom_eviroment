from Enviroment import Enviroment
import pygame
import time
import random
import matplotlib.pyplot as plt

obstacle_turn = False
vizualaze = True
Total_war = True
done = True


Vizual = False

env = Enviroment(obstacle_turn, vizualaze, Total_war)
state = env.reset()

totalRew = 0
for i in range(100):
    while done:

        action = (random.randint(10, 50), random.uniform(-0.2, 0.2))
        state, reward, done, numstep = env.step(action)
        totalRew += reward
        print("Шаг: {2} Награда: {0}, Состояние: {1}, Положение РТК: {4} Положение цели: {5} Действие: {3}" \
              .format(reward, done, numstep, action, state.posRobot, state.target, totalRew))

        if Vizual:
            plt.imshow(state.img)
            plt.pause(0.01)
    done = True
    # env.draw()
    if reward == 100:
        plt.imshow(state.img)
        plt.pause(0.01)
    state = env.reset()
    totalRew = 0
