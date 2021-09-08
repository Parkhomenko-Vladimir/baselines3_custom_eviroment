import torch as th
import torch.nn as nn
import torch.nn.functional as F
import gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor



class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 2310):
        """
        Инициализация нейронки
        :param observation_space: (gym.Space) Размер входного изображения
        :param features_dim: (int) Количество признаков на выходе свертки (количество входов во net_arch).
        """
        super(CustomCNN, self).__init__(observation_space, features_dim)
        extractors = {}
        
        for key, subspace in observation_space.spaces.items():
            if key == "img":
        
                n_input_channels = observation_space[key].shape[0]
            
                extractors[key] = nn.Sequential(

                nn.Conv2d(n_input_channels, 32, 2),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(32, 64, 2),
                nn.MaxPool2d(2, 2),

                ResBlock(n_filters=64, kernel_size=2),
                nn.MaxPool2d(4, 4),
                ResBlock(n_filters=64, kernel_size=2),
                nn.MaxPool2d(2, 2),
                ResBlock(n_filters=64, kernel_size=2),
                nn.MaxPool2d(2, 2),
                ResBlock(n_filters=64, kernel_size=2), 
                nn.MaxPool2d(2, 2),
                
                nn.Conv2d(64, 128, 2),
                nn.Flatten())
                    
            elif key == "posRobot":
                n_input_channels = observation_space[key].shape[0]
                            
                extractors[key] = nn.Sequential(nn.Linear(n_input_channels, 9),
                                        nn.ReLU(),
                                        nn.Linear(9, 9),
                                        nn.ReLU(),
                                        nn.Linear(9, 3))
                    
            elif key == "target":
                n_input_channels = observation_space[key].shape[0]
                            
                extractors[key] = nn.Sequential(nn.Linear(n_input_channels, 9),
                                        nn.ReLU(),
                                        nn.Linear(9, 9),
                                        nn.ReLU(),
                                        nn.Linear(9, 3))
                
        self.extractors = nn.ModuleDict(extractors)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        '''
        Forward propagation
        :param observations: (dict) изображение; координаты и углы ориентации агентов
        :return: features tensor
        '''
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))

        return th.cat(encoded_tensor_list, dim=1)


    
class ResBlock(nn.Module):
    def __init__(self, n_filters, kernel_size):
        """
        Инициализация кастомного резнетовского блока
        :param n_filters: (int) количество фильтров сверточного слоя
        :param kernel_size: (int) размер ядра свертки
        """
        super().__init__()
        self.n_filters = n_filters
        self.kernel_size = kernel_size

        self.b1 = nn.Conv2d(self.n_filters, self.n_filters, self.kernel_size, padding='same')
    
        self.b2 = nn.BatchNorm2d(self.n_filters, eps = 0.001, momentum= 0.99)
        self.b3 = nn.Conv2d(self.n_filters, self.n_filters, self.kernel_size, padding='same')
        self.b4 = nn.BatchNorm2d(self.n_filters, eps = 0.001, momentum= 0.99)
        
    def forward(self, x):
        '''
        Forward propagation
        :param x: input
        :return: output
        '''
        residual = x
        y = F.relu(self.b1(x))
        y = self.b2(y)
        y = F.relu(self.b3(y))
        y = self.b4(y)
        y += residual
        y = F.relu(y)
        return y