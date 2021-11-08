from Enviroment import Enviroment

import matplotlib.pyplot as plt
import keyboard

obstacle_turn = True
vizualaze = True
Total_war = True
done = False
head_velocity = 0.4    # скорость повората башни в радианах
num_obs = 40    # количество препятствий
size_obs = [30, 100]     # размер препятствий
m_step = 500    # максимальное количество шагов
num_enemy = 1   # количество противников

Vizual = False


for ist in range(100):
    env = Enviroment(obstacle_turn, vizualaze, Total_war, head_velocity, num_obs, num_enemy, size_obs, m_step)
    state = env.reset()
    #plt.imshow(state.img)
    plt.pause(1.5) # center_obs - массив препятствий [0] - координата по Х, [1] - координата по Y, [2] - ширина, [3] - высота


for i in range(120):
    while not done:

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
        # plt.imshow(state.img)
        plt.pause(0.01)
    state = env.reset()
