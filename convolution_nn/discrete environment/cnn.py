import gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn
import torch.nn.functional as F



class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 1024):
        """
        Инициализация нейронки
        :param observation_space: (gym.Space) Размер входного изображения
        :param features_dim: (int) Количество признаков на выходе свертки (количество входов во net_arch).
        """
        super(CustomCNN, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(

            nn.Conv2d(n_input_channels, 32, 2),
            nn.Conv2d(32, 64, 2),
            nn.MaxPool2d(2, 2),

            ResBlock(n_filters=64, kernel_size=2),
            nn.MaxPool2d(2, 2),

            ResBlock(n_filters=64, kernel_size=2),
            nn.MaxPool2d(2, 2),

            ResBlock(n_filters=64, kernel_size=2),
            ResBlock(n_filters=64, kernel_size=2),
            ResBlock(n_filters=64, kernel_size=2),
            nn.MaxPool2d(2, 2),

            ResBlock(n_filters=64, kernel_size=2),
            ResBlock(n_filters=64, kernel_size=2),
            ResBlock(n_filters=64, kernel_size=2),
            nn.MaxPool2d(2, 2),

            ResBlock(n_filters=64, kernel_size=2),
            nn.MaxPool2d(2, 2),

            ResBlock(n_filters=64, kernel_size=2),

            nn.Conv2d(64, 64, 4),
            nn.Flatten()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        '''
        Forward propagation
        :param observations: изображение на входе
        :return: features tensor
        '''
        return self.cnn(observations)


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
        self.b2 = nn.BatchNorm2d(self.n_filters, eps=0.001, momentum=0.99)
        self.b3 = nn.Conv2d(self.n_filters, self.n_filters, self.kernel_size, padding='same')
        self.b4 = nn.BatchNorm2d(self.n_filters, eps=0.001, momentum=0.99)

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

