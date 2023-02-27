import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

class PPOMemory:
    def __init__(self):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.a_states = []
        self.e_states = []

    def generate_butches(self):

        return np.array(self.states), \
               np.array(self.a_states), \
               np.array(self.e_states), \
               np.array(self.actions), \
               np.array(self.probs), \
               np.array(self.vals), \
               np.array(self.rewards), \
               np.array(self.dones), \

    def store_memory(self, state, a_state, e_state, action,\
                     probs, vals, reward, done):
        self.states.append(state)
        self.a_states.append(a_state)
        self.e_states.append(e_state)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.actions.append(action)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.a_states = []
        self.e_states = []

class ActorNetwork(nn.Module):
    def __init__(self, input_conv_dims, allias_state, enemy_state, n_actions,\
                 alpha, chkpt_dir='tmp/ppo'):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_ppo')
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

        self.l1 = nn.Linear(conv_out + 8 * 2, 128)
        self.l2 = nn.Linear(128, 64)
        self.lstm = nn.LSTMCell(64, 64)
        self.l3 = nn.Linear(64, 32)
        self.pi = nn.Linear(32, n_actions)
        self.st = nn.Softmax(dim=-1)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
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

        x = F.elu(self.l1(x))
        x = F.elu(self.l2(x))
        hx, cx = self.lstm(x, (hx, cx))
        x = F.elu(self.l3(hx))
        out = self.st((self.pi(x)))
        dist = Categorical(out)

        return dist, hx, cx

    def forward_for_butch(self, state, E_coords, A_coords,  hx, cx):
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

        x = F.elu(self.l1(x))
        x = F.elu(self.l2(x))
        hx, cx = self.lstm(x, (hx, cx))
        x = F.elu(self.l3(hx))
        x = self.st((self.pi(x)))
        # dist = Categorical(x)

        return x, hx, cx

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class CriticNetwork(nn.Module):
    def __init__(self, input_conv_dims, allias_state, enemy_state, \
                 alpha,  chkpt_dir='tmp/ppo'):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_ppo')
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

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
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

        x = F.elu(self.l1(x))
        x = F.elu(self.l2(x))
        x = F.elu(self.l3(x))
        value = self.V(x)

        return value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

def weights_init_uniform(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # apply a uniform distribution to the weights and a bias=0
        m.weight.data.uniform_(0.0, 1.0)
        m.bias.data.fill_(0)

class Agent:
    def __init__(self, input_conv_dims ,allias_state, enemy_state, n_actions, \
                 gamma=0.99, alpha=0.003, gae_lambda=0.95,
                 policy_clip=0.2, n_epochs=10):
        self.input_dim = input_conv_dims
        self.allias_state = allias_state
        self.enemy_state = enemy_state
        self.n_actions = n_actions

        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.n_epochs = n_epochs

        self.actor = ActorNetwork(input_conv_dims = self.input_dim,
                                  allias_state=allias_state,
                                  enemy_state=enemy_state,
                                  n_actions=n_actions,
                                  alpha=alpha)
        self.actor.apply(weights_init_uniform)

        self.critic = CriticNetwork(input_conv_dims = self.input_dim,
                                    allias_state=allias_state,
                                    enemy_state=enemy_state,
                                    alpha=alpha)
        # self.critic.apply(weights_init_uniform)

        self.memory = PPOMemory()

    def remember(self, state, a_state, e_state, action,\
                 probs, vals, reward, done):
        self.memory.store_memory(state, a_state, e_state, action,\
                                 probs, vals, reward, done)

    def butch_predict(self, state_sec, e_st, a_st, h ,c):

        m = []
        for i, e, a in zip(state_sec, e_st, a_st):
            x, h, c = self.actor.forward_for_butch(i, e, a, h ,c)
            m.append(x)
        dist = Categorical(T.stack(m))
        return dist

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation, hx, cx):

        img = T.tensor(observation['img'], dtype=T.float).to(self.actor.device)
        a = T.tensor(observation['posRobot'], dtype=T.float).to(self.actor.device)
        e = T.tensor(observation['target'], dtype=T.float).to(self.actor.device)

        dist, hidden, c = self.actor(state = img, E_coords = e, A_coords = a, hx = hx, cx = cx)
        value = self.critic(state = img, E_coords = e, A_coords = a)
        action = dist.sample()

        probs = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()

        return action, probs, value, hidden, c

    def learn(self, h, c):
        for i in range(self.n_epochs):
            state_arr, a_arr, e_arr ,action_arr, old_probs_arr, vals_arr, \
            reward_arr, done_arrs = self.memory.generate_butches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (reward_arr[k] + self.gamma * values[k + 1] * \
                                       (1 - int(done_arrs[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t
            advantage = T.tensor(advantage).to(self.actor.device)

            values = T.tensor(values).to(self.actor.device)

            states = T.tensor(state_arr, dtype=T.float).to(self.actor.device)
            a = T.tensor(a_arr, dtype=T.float).to(self.actor.device)
            e = T.tensor(e_arr, dtype=T.float).to(self.actor.device)

            old_probs = T.tensor(old_probs_arr).to(self.actor.device)
            actions = T.tensor(action_arr).to(self.actor.device)

            dist = self.butch_predict(states, e, a, h ,c)

            critic_value = self.critic(states.squeeze(1), e.squeeze(1), a.squeeze(1))
            critic_value = T.squeeze(critic_value)

            new_probs = dist.log_prob(actions)
            # prob_ratio = new_probs.exp() / old_probs.exp()
            prob_ratio = (new_probs - old_probs).exp()
            weight_probs = advantage * prob_ratio
            weight_clipped_probs = T.clamp(prob_ratio, 1 - self.policy_clip,
                                           1 + self.policy_clip) * advantage
            actor_loss = -T.min(weight_probs, weight_clipped_probs).mean()

            returns = advantage + values
            critic_loss = (returns - critic_value) ** 2
            critic_loss = critic_loss.mean()

            total_loss = actor_loss + 0.5 * critic_loss
            self.actor.optimizer.zero_grad()
            self.critic.optimizer.zero_grad()
            total_loss.backward()
            self.actor.optimizer.step()
            self.critic.optimizer.step()

        self.memory.clear_memory()

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
