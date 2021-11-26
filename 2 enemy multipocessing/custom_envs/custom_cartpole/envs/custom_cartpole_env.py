import os
import gym
import numpy as np

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.callbacks import BaseCallback

from Enviroment import Enviroment
from gym import spaces
import cv2
from tqdm import tqdm

from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCartPoleEnv(gym.Env):
    '''
    Оборочивание класса среды в среду gym
    '''
    metadata = {'render.modes': ['human']}

    def __init__(self, obstacle_turn: bool, Total_war: bool,
                 num_obs: int, num_enemy: int, num_alias: int, 
                 size_obs, steps_limit, vizualaze=False, head_velocity=0.01):
        '''
        Инициализация класса среды
        :param obstacle_turn: (bool) Флаг генерации препятствий
        :param vizualaze: (bool) Флаг генерации препятствий
        :param Total_war: (bool) Флаг режима игры (с противником или без)
        :param steps_limit: (int) Максимальное количество действий в среде за одну игру
        '''
        self.log_koef = 50

        self.velocity_coef = 35       #  1/2 max speed !!!
        self.ang_Norm_coef = np.pi
        self.coords_Norm_coef = 500
        self.n_alias = num_alias
        self.num_enemies = num_enemy
        
        self.enviroment = Enviroment(obstacle_turn, vizualaze, Total_war, head_velocity, num_obs, num_enemy,
                         num_alias, size_obs, steps_limit)

        state = self.enviroment.reset()

        self.action_space = spaces.Box(low=np.array([-1, -1,-1, -1]), high=np.array([1, 1, 1, 1]), dtype=np.float16)
        self.observation_space = gym.spaces.Dict({
                    'img': spaces.Box(low=0, high=255, shape=(500, 500, 3), dtype=np.uint8),
                    'posRobot': spaces.Box(low  = np.array([[0, 0,-1] for i in range(num_alias)]).reshape(-1), 
                                           high = np.array([[1, 1, 1] for i in range(num_alias)]).reshape(-1),
                                           dtype = np.float32),
                    'target': spaces.Box(low  = np.array([[0, 0,-1] for i in range(num_enemy)]).reshape(-1), 
                                         high = np.array([[1, 1, 1] for i in range(num_enemy)]).reshape(-1),
                                         dtype = np.float32)
                                                })
        
        self.a_f = False
        self.e_f = False
        
        self.changecoordsA = lambda x: x
        self.changecoordsE = lambda x: x
        
        self.img1 = state.img
        self.img2 = state.img
        self.img3 = state.img
        self.Img = None
        

    def make_layers(self):
        """
        Функция наслоения изображений трех последовательных шагов в среде
        :param img1, img2, img3: состояния среды на трех последовательных шагах
        :return: new_img: изображение, содержащее информацию о состояниях среды на трех последовательных шагах, отображенную с разной интенсивностью
        """
        new_img = cv2.addWeighted(self.img2, 0.4, self.img1, 0.2, 0)
        self.Img = cv2.addWeighted(self.img3, 0.7, new_img, 0.5, 0)
    
    
    def step(self, action):
        """
        Метод осуществления шага в среде
        :param action: (int) направление движения в среде
        :return: dict_state, reward, not done, {}: состояние, реворд, флаг терминального состояния, информация о среде
        """
        
        action[0] *= self.velocity_coef
        action[1] *= self.ang_Norm_coef
        
        action[2] *= self.velocity_coef
        action[3] *= self.ang_Norm_coef
        
        action = np.array([action[:2],
                          action[2:]])
        
        state, reward, done, numstep = self.enviroment.step(action)
        
        self.img1 = self.img2
        self.img2 = self.img3
        self.img3 = state.img
        
        self.make_layers()
                
        # координаты союзника
        
        if bool(state.posRobot[0,3]) and not self.a_f:
            self.a_f = True
            self.changecoordsA = self.FirstCtoSecondC
        
        elif bool(state.posRobot[1,3]) and not self.a_f:
            self.a_f = True
            self.changecoordsA = self.SecondCtoFirstC
        
#         коордтнаты противника

        if bool(state.target[0,3]) and not self.e_f:
            self.e_f = True
            self.changecoordsE = self.FirstCtoSecondC
        
        elif bool(state.target[1,3]) and not self.e_f:
            self.e_f = True
            self.changecoordsE = self.FirstCtoSecondC
        
        self.changecoordsE(state.target)
        self.changecoordsA(state.posRobot)
        
        Enemy_coord = state.target
        Alias_coord = state.posRobot
        
        Enemy_coord = Enemy_coord[:,:3]
        Alias_coord = Alias_coord[:,:3]
    
        dist = np.sqrt((np.array(Enemy_coord[:, 0]) - np.array(Alias_coord[:, 0]).reshape(self.n_alias, 1)) ** 2 \
                     + (np.array(Enemy_coord[:, 1]) - np.array(Alias_coord[:, 1]).reshape(self.n_alias, 1)) ** 2)
        
        Ax = np.cos(Enemy_coord[:, 2])
        Ay = np.sin(Enemy_coord[:, 2])
        Bx = np.array(Alias_coord[:, 0]) - np.array(Enemy_coord[:, 0]).reshape(self.num_enemies, 1)
        By = np.array(Alias_coord[:, 1]) - np.array(Enemy_coord[:, 1]).reshape(self.num_enemies, 1)
        phy = np.arccos((Ax * Bx + Ay * By) / (np.sqrt(Ax ** 2 + Ay ** 2) * np.sqrt(Bx ** 2 + By ** 2)))
        phy[(phy == 0) | (np.isnan(phy))] = 1e-6
        reward = np.sum(reward) + np.sum(np.log2(phy / dist * 50))
        
        dict_state = {'img':     self.Img,  
                      'posRobot':self.norm(Alias_coord).reshape(-1),  
                      'target':  self.norm(Enemy_coord).reshape(-1)}

        return dict_state, reward, done, {}
    
    def norm(self, coords):
        '''
        Метод нормализации координат
        :return: coords: нормализованные координаты
        '''
        coords=np.float32(coords)
        coords[:,2]  = coords[:,2] / self.ang_Norm_coef #угол
        coords[:,:2] = coords[:,:2] / self.coords_Norm_coef #координаты
        
        return coords


    def reset(self):
        '''
        Метод обновления игры
        :return: dict_state: состояние
        '''
        
        state = self.enviroment.reset()
        
        self.img2 = state.img
        self.img3 = state.img
        
        state.posRobot = state.posRobot[:,:3]
        state.target = state.target[:,:3]
        
        dict_state = {'img':     state.img,  
                      'posRobot':self.norm(state.posRobot).reshape(-1),  
                      'target':  self.norm(state.target).reshape(-1)}

        return dict_state
    
    def FirstCtoSecondC(self,x):
        x[0,:3] = x[1,:3]
        return x
        
    def SecondCtoFirstC(self,x):
        x[1,:3] = x[0,:3]
        return x

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
                obs, _, done ,_ = self.step(action)
                img = obs['img']
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                out.write(img)
            out.release()
    
    def get_statistic(self, model, num_games):
        collision = 0
        win = 0
        destroyed = 0
        loss = 0
        
        pbar = tqdm(range(num_games))
        for i in pbar:
            obs = self.reset()
            done = False
            while not done:
                action, _ = model.predict(obs)
                obs, reward, done ,_ = self.step(action)   
                
            if reward == -30:#win
                collision+=1
            elif reward == 100:# loss
                win +=1
            elif reward == -100:# loss
                destroyed +=1
            else:    #not_achieved
                loss+=1
        
        print("Win: ",win/num_games)
        print("destroyed: ", destroyed/num_games)
        print("loss: ",loss/num_games)
        print("collision: ",collision/num_games)
