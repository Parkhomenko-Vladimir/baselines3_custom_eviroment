import numpy as np
import gym
from gym import spaces
from Enviroment import Enviroment


class CustomEnv(gym.Env):
    '''
    Оборочивание класса среды в среду gym
    '''
    metadata = {'render.modes': ['human']}

    def __init__(self, obstacle_turn: bool, Total_war: bool, vizualaze=False, head_velosity=0.01):
        '''
        Инициализация класса среды
        :param obstacle_turn: (bool) Флаг генерации препятствий
        :param vizualaze: (bool) Флаг генерации препятствий
        :param Total_war: (bool) Флаг режима игры (с противником или без)
        :param steps_limit: (int) Максимальное количество действий в среде за одну игру
        '''
        super(CustomEnv, self).__init__()

        self.env1 = Enviroment(obstacle_turn, vizualaze, Total_war, head_velosity)

        state = self.env1.reset()
        
        self.action_space = spaces.Box(low=np.array([-0.0000001, -3.14]), high=np.array([70, 3.14]), dtype=np.float16)
        self.observation_space = gym.spaces.Dict({
        'img': spaces.Box(low=0, high=255, shape=(500, 500, 3), dtype=np.uint8),
        'posRobot': spaces.Box(low=np.array([0, 0, -3.14]), high=np.array([500, 500, 3.14])),
        'target': spaces.Box(low=np.array([0, 0, -3.14]), high=np.array([500, 500, 3.14]))})  # 

    def step(self, action):
        """
        Метод осуществления шага в среде
        :param action: (int) направление движения в среде
        :return: dict_state, reward, not done, {}: состояние, реворд, флаг терминального состояния, информация о среде
        """
        state, reward, done, numstep = self.env1.step(action)
        dict_state = {'img':     state.img,
                      'posRobot':state.posRobot,
                      'target':  state.target}
#         action[0] = 70*action[0]
        return dict_state, reward, done, {}

    def reset(self):
        '''
        Метод обновления игры
        :return: dict_state: состояние
        '''
        state = self.env1.reset()
        dict_state = {'img':     state.img,
                      'posRobot':state.posRobot,
                      'target':  state.target}
        
        return dict_state

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