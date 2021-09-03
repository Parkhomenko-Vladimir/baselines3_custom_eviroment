import gym
from gym import spaces
from Enviroment import Enviroment
import numpy as np

class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, obstacle_turn = False, vizualaze = False, Total_war = True, steps_limit = 1000):
        super(CustomEnv, self).__init__()

#         obstacle_turn = False
#         vizualaze = True
#         Total_war = True
        self.steps_limit = steps_limit
        self.env1 = Enviroment(obstacle_turn, vizualaze, Total_war)

        state = self.env1.reset()

        self.action_space = spaces.Discrete(8)
        
        self.observation_space = gym.spaces.Dict({
                    'img': spaces.Box(low=0, high=255, shape=(500, 500, 3), dtype=np.uint8),
                    'posRobot': spaces.Box(low=np.array([0, 0,-3.14]), high=np.array([500, 500, 3.14])),
#                     'posRobot': spaces.Box(low=-3.14, high=500, shape=((3,))),
                    'target': spaces.Box(low=0, high=500, shape=((2,)))
                    })

        
    def step(self, action):

        state, reward, done, numstep = self.env1.step(action)
#         observation , reward, done = state ,reward , done
        
        dict_state = {'img':     state.img,
                      'posRobot':state.posRobot,
                      'target':  state.target}
        
        if numstep >= self.steps_limit:
            done = False
    
        return dict_state, reward, not done, {}


    def reset(self):

        state = self.env1.reset()
        
        dict_state = {'img':     state.img,  # np.array
                      'posRobot':state.posRobot,  # list
                      'target':  state.target}  # list
        
        return dict_state  

    def render(self, mode='human'):
        pass