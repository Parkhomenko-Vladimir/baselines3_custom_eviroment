import numpy as np
import gym
from gym import spaces
from Enviroment import Enviroment


class CustomEnv(gym.Env):
    '''
    Оборочивание класса среды в среду gym
    '''
    metadata = {'render.modes': ['human']}

    def __init__(self, obstacle_turn: bool, vizualaze=False, Total_war=False):
        '''
        Инициализация класса среды
        :param obstacle_turn: (bool) Флаг генерации препятствий
        :param vizualaze: (bool) Флаг генерации препятствий
        :param Total_war: (bool) Флаг режима игры (с противником или без)
        :param steps_limit: (int) Максимальное количество действий в среде за одну игру
        '''
        super(CustomEnv, self).__init__()

        self.env1 = Enviroment(obstacle_turn, vizualaze, Total_war)

        state = self.env1.reset()

        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(low=0, high=255, shape=state.img.shape, dtype=np.uint8)

    def step(self, action):
        """
        Метод осуществления шага в среде
        :param action: (int) направление движения в среде
        :return: observation, reward, not done, {}: состояние, реворд, флаг терминального состояния, информация о среде
        """
        state, reward, done, numstep = self.env1.step(action)
        observation, reward, done = state.img, reward, done

        return observation, reward, done, {}

    def reset(self):
        '''
        Метод обновления игры
        :return: state: состояние
        '''

        state = self.env1.reset()

        return state.img

    def render(self, model, num_gifs=1):
        '''
        Метод вывода информации об игре
        :param mode:
        :return:
        '''
        for i in range(num_gifs):
            images = []
            obs = self.reset()
            img = obs['img']  # env.render(mode='rgb_array')
            done = True
            while done:
                images.append(img)
                action = model.predict(obs)
                obs, _, done ,_ = self.step(int(action[0]))
                img = obs['img']  # env.render(mode='rgb_array')

            imageio.mimsave(f"video{i}.gif", [np.array(img) for i, img in enumerate(images) if i%2 == 0], fps=30)