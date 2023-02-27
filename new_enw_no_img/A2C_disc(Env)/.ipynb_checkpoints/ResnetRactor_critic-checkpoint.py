import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class CoordModel(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(CoordModel, self).__init__()

        self.l1 = nn.Linear(input_dim, 32)
        self.l2 = nn.Linear(32, 32)
        self.l3 = nn.Linear(32, out_dim)

    def forward(self, x):
        x = T.relu(self.l1(x))
        x = T.relu(self.l2(x))
        x = T.relu(self.l3(x))

        return x

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

class ActorCritic():
    def __init__(self, input_conv_dims,allias_state, enemy_state, n_actions,\
                 lr , gamma, tau = 1.0):
        self.input_dim = input_conv_dims
        self.allias_state = allias_state
        self.enemy_state = enemy_state
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.tau = tau

        self.actor = Actor(input_conv_dims = self.input_dim,
                           allias_state=allias_state,
                           enemy_state=enemy_state,
                             n_actions = self.n_actions,
                             lr = self.lr)
        self.critic = Critic(input_conv_dims = self.input_dim,
                             allias_state=allias_state,
                             enemy_state=enemy_state,
                             lr = self.lr)

        self.reward_mem   = []
        self.log_prob_mem = []
        self.values_mem   = []

    def store_mem(self, reward):
        self.reward_mem.append(reward)
        # self.log_prob_mem.append(log_prob)
        # self.values_mem.append(value)

    def sample_memory(self):
        return self.reward_mem,\
               self.log_prob_mem,\
               self.values_mem

    def clear_memory(self):
        self.reward_mem = []
        self.log_prob_mem = []
        self.values_mem = []

    def choose_action(self, state, hx, cx):

        img = T.tensor(state['img'], dtype=T.float).to(self.actor.device)
        a = T.tensor(state['posRobot'], dtype=T.float).to(self.actor.device)
        e = T.tensor(state['target'], dtype=T.float).to(self.actor.device)


        action, log_prob, hx, cx = self.actor(state = img,
                                              E_coords = e,
                                              A_coords = a,
                                              hx = hx,
                                              cx = cx)
        value = self.critic(state = img,
                            E_coords = e,
                            A_coords = a)

        self.log_prob_mem.append(log_prob)
        self.values_mem.append(value)

        return action.item(), hx, cx
    
    def choose_action_test(self, state, hx, cx):

        img = T.tensor(state['img'], dtype=T.float).to(self.actor.device)
        a = T.tensor(state['posRobot'], dtype=T.float).to(self.actor.device)
        e = T.tensor(state['target'], dtype=T.float).to(self.actor.device)


        action, log_prob, hx, cx = self.actor(state = img,
                                              E_coords = e,
                                              A_coords = a,
                                              hx = hx,
                                              cx = cx)
        value = self.critic(state = img,
                            E_coords = e,
                            A_coords = a)

        # self.log_prob_mem.append(log_prob)
        # self.values_mem.append(value)

        return action.item(), hx, cx

    def calc_R(self, done, rewards, values):

        values = T.cat(values).squeeze()

        if len(values.size()) == 1:
            R = values[-1]*(1-int(done))
        elif len(values.size()) == 0:
            R = values * (1 - int(done))

        batch_return = []
        for reward in rewards[::-1]:
            R = reward + self.gamma * R
            batch_return.append(R)
        batch_return.reverse()
        batch_return = T.tensor(batch_return,
                                dtype=T.float).reshape(values.size()).to(self.actor.device)
        return batch_return

    def learn(self, new_state,done):
        rewards, log_probs, values = self.sample_memory()

        returns = self.calc_R(done, rewards, values)

        img = T.tensor(new_state['img'], dtype=T.float).to(self.actor.device)
        a = T.tensor(new_state['posRobot'], dtype=T.float).to(self.actor.device)
        e = T.tensor(new_state['target'], dtype=T.float).to(self.actor.device)

        next_v = T.zeros(1, 1).to(self.actor.device) if done else \
                    self.critic(state=img, E_coords=e, A_coords=a)

        values.append(next_v.detach())
        values = T.cat(values).squeeze()
        log_probs = T.cat(log_probs)
        rewards = T.tensor(rewards).to(self.actor.device)

        delta_t = rewards + self.gamma * values[1:] - values[:-1]
        n_steps = len(delta_t)
        gae = np.zeros(n_steps)
        for t in range(n_steps):
            for k in range(0, n_steps - t):
                temp = (self.gamma * self.tau) ** k * delta_t[t + k]
                gae[t] += temp
        gae = T.tensor(gae, dtype=T.float).to(self.actor.device)

        entropy_loss = (-log_probs * T.exp(log_probs)).sum()
        actor_loss = -(log_probs * gae).sum() - 0.01 * entropy_loss
        critic_loss = F.mse_loss(values[:-1].squeeze(), returns)

        actor_loss.backward()
        critic_loss.backward()

        T.nn.utils.clip_grad_norm(self.actor.parameters(), 10)
        T.nn.utils.clip_grad_norm(self.critic.parameters(), 10)

        self.actor.optimizer.step()
        self.critic.optimizer.step()

        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()

        self.clear_memory()

        return actor_loss.item(), critic_loss.item()

    def save_models(self):
        print('.... saving models ....')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('.... loading models ....')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

class Actor(nn.Module):
    def __init__(self, input_conv_dims, allias_state, enemy_state,\
                 n_actions, lr = 0.001):
        super(Actor, self).__init__()

        self.checkpoint_file = 'tmp/Actor'
        self.input_dims = input_conv_dims
        self.allias_state = allias_state
        self.enemy_state = enemy_state

        self.conv1 = nn.Conv2d(input_conv_dims[0], 32, 3, stride=2, padding=1)
        self.pool1 = nn.MaxPool2d(4)
        self.conv2 = ResBlock(n_filters=32, kernel_size=4)
        self.pool2 = nn.MaxPool2d(4)
        self.conv3 = ResBlock(n_filters=32, kernel_size=4)
        self.pool3 = nn.MaxPool2d(4)
        self.conv4 = ResBlock(n_filters=32, kernel_size=4)
        self.flat = nn.Flatten(start_dim=1, end_dim=3)

        self.alias_layer = CoordModel(allias_state, 8)
        self.enemy_layer = CoordModel(enemy_state, 8)

        conv_out = self.calc_conv_output()

        self.l1 = nn.Linear(conv_out + 8*2, 128)
        self.l2 = nn.Linear(128, 64)
        self.lstm = nn.LSTMCell(64,64)
        self.l3 = nn.Linear(64,32)
        self.pi = nn.Linear(32, n_actions)

        self.optimizer = T.optim.RMSprop(self.parameters(), lr=lr)
        self.device = T.device('cuda:1' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def calc_conv_output(self):
        state = T.zeros(1, *self.input_dims)

        dims = self.conv1(state)
        dims = self.pool1(dims)
        dims = self.conv2(dims)
        dims = self.pool2(dims)
        dims = self.conv3(dims)
        dims = self.pool3(dims)
        dims = self.conv4(dims)
        conv = self.flat(dims)
        return conv.shape[1]

    def forward(self, state, E_coords, A_coords,  hx, cx):

        conv = self.conv1(state)
        conv = self.pool1(conv)
        conv = self.conv2(conv)
        conv = self.pool2(conv)
        conv = self.conv3(conv)
        conv = self.pool3(conv)
        conv = self.conv4(conv)
        conv = self.flat(conv)

        a = self.alias_layer(A_coords)
        e = self.enemy_layer(E_coords)

        x = T.cat((conv, a, e), 1)

        x = F.tanh(self.l1(x))
        x = F.tanh(self.l2(x))
        hx, cx = self.lstm(x, (hx, cx))
        x = F.tanh(self.l3(hx))
        out = self.pi(x)

        probs = T.softmax(out, dim=1)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action[0], log_prob, hx, cx

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class Critic(nn.Module):
    def __init__(self, input_conv_dims, allias_state, enemy_state, lr = 0.001):
        super(Critic, self).__init__()

        self.checkpoint_file = 'tmp/Critic'
        self.input_dims = input_conv_dims

        self.conv1 = nn.Conv2d(input_conv_dims[0], 32, 3, stride=2, padding=1)
        self.pool1 = nn.MaxPool2d(4)
        self.conv2 = ResBlock(n_filters=32, kernel_size=4)
        self.pool2 = nn.MaxPool2d(4)
        self.conv3 = ResBlock(n_filters=32, kernel_size=4)
        self.pool3 = nn.MaxPool2d(4)
        self.conv4 = ResBlock(n_filters=32, kernel_size=4)
        self.flat = nn.Flatten(start_dim=1, end_dim=3)

        self.alias_layer = CoordModel(allias_state, 8)
        self.enemy_layer = CoordModel(enemy_state, 8)

        conv_out = self.calc_conv_output()

        self.l1 = nn.Linear(conv_out + 8 * 2, 128)
        self.l2 = nn.Linear(128, 64)
        self.l3 = nn.Linear(64, 32)
        self.V = nn.Linear(32, 1)

        self.optimizer = T.optim.RMSprop(self.parameters(), lr=lr)
        self.device = T.device('cuda:1' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def calc_conv_output(self):
        state = T.zeros(1, *self.input_dims)

        dims = self.conv1(state)
        dims = self.pool1(dims)
        dims = self.conv2(dims)
        dims = self.pool2(dims)
        dims = self.conv3(dims)
        dims = self.pool3(dims)
        dims = self.conv4(dims)
        conv = self.flat(dims)
        return conv.shape[1]

    def forward(self, state, E_coords, A_coords):
        conv = self.conv1(state)
        conv = self.pool1(conv)
        conv = self.conv2(conv)
        conv = self.pool2(conv)
        conv = self.conv3(conv)
        conv = self.pool3(conv)
        conv = self.conv4(conv)
        conv = self.flat(conv)

        a = self.alias_layer(A_coords)
        e = self.enemy_layer(E_coords)

        x = T.cat((conv, a, e), 1)

        x = F.tanh(self.l1(x))
        x = F.tanh(self.l2(x))
        x = F.tanh(self.l3(x))
        value = self.V(x)

        return value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

# ac = Actor(input_dims = (3,500,500), n_actions = 5)
# cr = Critic(input_dims = (3,500,500))