from Enviroment import Enviroment
import pygame
import time
import random
import matplotlib.pyplot as plt
import keyboard

obstacle_turn = False
vizualaze = True
Total_war = True

done = False
head_velocity = 0.00     #скорость повората башни в радианах
Vizual = False


env = Enviroment(obstacle_turn, vizualaze, Total_war, head_velocity)
state = env.reset()


for i in range(120):
    while not done:
        # time.sleep(0.1)
        #action = random.randint(0, 7)
        # action = input("uio")
        keyboard.read_key()
        if keyboard.is_pressed('6'):
            action = 0
        elif keyboard.is_pressed('9'):
            action = 1
        elif keyboard.is_pressed('8'):
            action = 2
        elif keyboard.is_pressed('7'):
            action = 3
        elif keyboard.is_pressed('4'):
            action = 4
        elif keyboard.is_pressed('1'):
            action = 5
        elif keyboard.is_pressed('2'):
            action = 6
        elif keyboard.is_pressed('3'):
            action = 7
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
