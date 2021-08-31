from Enviroment import Enviroment
import pygame
import time
import random
import matplotlib.pyplot as plt

obstacle_turn = False
vizualaze = True
Total_war = True

done = False

Vizual = False


env = Enviroment(obstacle_turn, vizualaze, Total_war)
state = env.reset()


for i in range(120):
    while not done:
        # time.sleep(0.1)
        action = random.randint(0, 7)

        state, reward, done, numstep = env.step(action)
        print("Шаг: {2} Награда: {0}, Состояние: {1}, Положение РТК: {4} Положение цели: {5} Действие: {3}"\
              .format(reward, done, numstep, action, state.posRobot, state.target))

        if Vizual:
            plt.imshow(state.img)
            plt.pause(0.01)
    done = False
    # env.draw()
    if reward == 100:
        plt.imshow(state.img)
        plt.pause(0.01)
    state = env.reset()
