import numpy as np
import gym
from gym import spaces
from Enviroment import Enviroment

class CustomEnv(gym.Env):
    '''
    Оборочивание класса среды в среду gym
    '''
    metadata = {'render.modes': ['human']}

    def __init__(self, obstacle_turn: bool, Total_war: bool, num_obs: int, num_enemy: int, size_obs, steps_limit, vizualaze=False, head_velocity=0.01):
        '''
        Инициализация класса среды
        :param obstacle_turn: (bool) Флаг генерации препятствий
        :param vizualaze: (bool) Флаг генерации препятствий
        :param Total_war: (bool) Флаг режима игры (с противником или без)
        :param steps_limit: (int) Максимальное количество действий в среде за одну игру
        '''

        self.velocity = 70
        self.log_koef = 50

        self.enviroment = Enviroment(obstacle_turn, vizualaze, Total_war, head_velocity, num_obs, num_enemy, size_obs, steps_limit)

        state = self.enviroment.reset()

        self.action_space = spaces.Box(low=np.array([-0.1, -3.14]), high=np.array([1, 3.14]), dtype=np.float16)
        self.observation_space = gym.spaces.Dict({
                    'img': spaces.Box(low=0, high=255, shape=(500, 500, 3), dtype=np.uint8),
                    'posRobot': spaces.Box(low=np.array([0, 0,-3.14]), high=np.array([500, 500, 3.14])),
                    'target': spaces.Box(low  = np.array([[0, 0,-3.14] for i in range(num_enemy)]).reshape(-1), 
                                         high = np.array([[500, 500, 3.14] for i in range(num_enemy)]).reshape(-1)
                                        )
                                                })

    def step(self, action):
        """
        Метод осуществления шага в среде
        :param action: (int) направление движения в среде
        :return: dict_state, reward, not done, {}: состояние, реворд, флаг терминального состояния, информация о среде
        """
        action[0] *= self.velocity
        state, reward, done, numstep = self.enviroment.step(action)
        
#         dict_state = {'img':     state.img,
#                       'posRobot':state.posRobot,
#                       'target':  state.target} 
#         for i in range(len(state.target)):
        
#             dist = np.sqrt((dict_state['target'][i][0]-dict_state['posRobot'][0])**2 + (dict_state['target'][i][1]-dict_state['posRobot'][1])**2)

#             Ax = np.cos(dict_state['target'][i][2])
#             Ay = -np.sin(dict_state['target'][i][2])
#             Bx = dict_state['posRobot'][0] - dict_state['target'][i][0]
#             By = dict_state['posRobot'][1] - dict_state['target'][i][1] 


#             phy = np.arccos((Ax*Bx + Ay*By)/(np.sqrt(Ax**2 + Ay**2) * np.sqrt(Bx**2 + By**2)))

#             reward = reward + np.log2(phy/dist*self.log_koef )
            
        dict_state = {'img':     state.img,
                      'posRobot':state.posRobot,
                      'target':  state.target.reshape(-1)}
        
        steps.append(dict_state['img'])
        if len(steps) > 3:
            steps.pop(0)
        if len(steps) < 3: # anyway, here are 2 items in list
            new_img = cv2.addWeighted(steps[1], 0.7, steps[0], 0.2, 0)
        else:
            new_img = cv2.addWeighted(steps[1], 0.4, steps[0], 0.2, 0)
            new_img = cv2.addWeighted(steps[2], 0.7, new_img, 0.5, 0)

        return dict_state, reward, done, {}

    def reset(self):
        '''
        Метод обновления игры
        :return: dict_state: состояние
        '''
        steps.clear()
        
        state = self.enviroment.reset()
        
        dict_state = {'img':     state.img,  
                      'posRobot':state.posRobot,  
                      'target':  state.target.reshape(-1)}
        
        steps.append(dict_state['img'])
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
            img = obs['img']# env.render(mode='rgb_array')
            done = False
                
            height, width, layers = img.shape
            size = (width,height)
            out = cv2.VideoWriter(f"video{i}.avi",cv2.VideoWriter_fourcc(*'DIVX'), 25, size)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            out.write(img)
            while not done:

                action, _ = model.predict(obs)
                print(action)
                obs, _, done ,_ = self.step(action)
                img = obs['img']
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                out.write(img)
            out.release()