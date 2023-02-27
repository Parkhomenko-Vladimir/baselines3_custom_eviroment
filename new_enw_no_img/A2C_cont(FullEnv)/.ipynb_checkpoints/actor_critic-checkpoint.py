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

class ActorCritic():
    def __init__(self, input_conv_dims, allias_state, enemy_state, n_actions,\
                 lr , gamma, max_actions, tau = 1.0):
        self.input_dim = input_conv_dims
        self.allias_state = allias_state
        self.enemy_state = enemy_state
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.tau = tau

        self.actor = Actor(input_conv_dims = self.input_dim,
                           allias_state = allias_state,
                           enemy_state = enemy_state,
                           n_actions = n_actions,
                           max_actions = max_actions,
                           lr = lr)
        self.critic = Critic(input_conv_dims = self.input_dim,
                           allias_state = allias_state,
                           enemy_state = enemy_state,
                           lr = lr)

        self.reward_mem   = []
        self.log_prob_mem = []
        self.values_mem   = []

    def store_mem(self, reward):
        self.reward_mem.append(reward)

    def sample_memory(self):
        return self.reward_mem,\
               self.log_prob_mem,\
               self.values_mem

    def clear_memory(self):
        self.reward_mem = []
        self.log_prob_mem = []
        self.values_mem = []

    def choose_action(self, state):

        img = T.tensor(state['img'], dtype = T.float).to(self.actor.device)
        a = T.tensor(state['posRobot'], dtype = T.float).to(self.actor.device)
        e = T.tensor(state['target'], dtype = T.float).to(self.actor.device)

        action, log_prob = self.actor(state = img, E_coords = e, A_coords = a)
        value = self.critic(state = img,E_coords = e, A_coords = a)

        self.log_prob_mem.append(log_prob)
        self.values_mem.append(value)

        return action.cpu().numpy()
    
    def choose_action_test(self, state):

        img = T.tensor(state['img'], dtype = T.float).to(self.actor.device)
        a = T.tensor(state['posRobot'], dtype = T.float).to(self.actor.device)
        e = T.tensor(state['target'], dtype = T.float).to(self.actor.device)

        action, log_prob = self.actor(state = img, E_coords = e, A_coords = a)
        value = self.critic(state = img,E_coords = e, A_coords = a)

        # self.log_prob_mem.append(log_prob)
        # self.values_mem.append(value)

        return action.cpu().numpy()

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

        next_v = T.zeros(1,1).to(self.actor.device) if done else \
                            self.critic(state = img,E_coords = e, A_coords = a)

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

        critic_loss = F.mse_loss(values[:-1].squeeze(), returns)
        entropy_loss = (-log_probs * T.exp(log_probs)).sum()

        actor_loss = -(log_probs * gae).sum() - 0.3 * entropy_loss

        actor_loss.backward()
        critic_loss.backward()

        # T.nn.utils.clip_grad_norm(self.actor.parameters(), 40)
        # T.nn.utils.clip_grad_norm(self.critic.parameters(), 40)

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
                 n_actions, max_actions, lr = 0.01):
        super(Actor, self).__init__()

        self.checkpoint_file = 'tmp/Actor'
        self.input_dims = input_conv_dims
        self.max_actions = max_actions
        self.allias_state = allias_state
        self.enemy_state = enemy_state

        self.conv1 = nn.Conv2d(in_channels=input_conv_dims[0], out_channels=8, kernel_size=8, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=4, stride=2, padding=0)
        self.pool = nn.AvgPool2d(kernel_size=4)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=0)
        self.norm = nn.BatchNorm2d(num_features = 32)
        self.flat = nn.Flatten(start_dim=1, end_dim= 3)

        self.alias_layer = CoordModel(allias_state, 8)
        self.enemy_layer = CoordModel(enemy_state, 8)

        conv_out = self.calc_conv_output()

        self.l1 = T.nn.Linear(conv_out + 8*2, 64)
        self.l2 = T.nn.Linear(64, 32)
        self.mu = T.nn.Linear(32, n_actions)
        self.sigma = T.nn.Linear(32, n_actions)

        self.optimizer = T.optim.RMSprop(self.parameters(), lr=lr)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def calc_conv_output(self):
        state = T.zeros(1, *self.input_dims)

        conv = self.conv1(state)
        conv = self.conv2(conv)
        conv = self.pool(conv)
        conv = self.conv3(conv)
        conv = self.norm(conv)
        conv = self.flat(conv)
        return conv.shape[1]

    def forward(self, state,E_coords, A_coords):

        conv = self.conv1(state)
        conv = self.conv2(conv)
        ########################################################
        # chA   = self.chanelAtention(conv)
        # poseA = self.poseAtention(conv)
        # conv = T.add(chA, conv)
        # conv = T.add(poseA, conv)
        ########################################################
        conv = self.pool(conv)
        conv = self.conv3(conv)
        ########################################################
        # chA   = self.chanelAtention(conv)
        # poseA = self.poseAtention(conv)
        # conv = T.add(chA, conv)
        # conv = T.add(poseA, conv)
        ########################################################
        # conv = self.norm(conv)
        conv = self.flat(conv)
        a = self.alias_layer(A_coords)
        e = self.enemy_layer(E_coords)

        x = T.cat((conv, a, e), 1)

        x = T.relu(self.l1(x))
        x = T.relu(self.l2(x))

        mu = 2 * T.tanh(self.mu(x))
        sigma = F.softplus(self.sigma(x)) + 0.001

        action_probs_d = T.distributions.Normal(mu, sigma)
        actions = action_probs_d.sample()

        log_probs = action_probs_d.log_prob(actions)
        log_probs = log_probs.sum(1, keepdim=True)

        return actions[0], log_probs[0]

    def chanelAtention(self, x):

        batch, ch, width, height = x.shape

        b =  x.reshape(batch, ch, width*height) # to [chanells, w*h]
        c =  x.reshape(batch, ch, width*height).mT # to [chanells, w*h].T
        d =  x.reshape(batch, ch, width*height)

        M_d = T.matmul(b,c)
        S = T.softmax(M_d, dim=1)
        out = T.matmul(S.mT,d).reshape(batch,ch, width, height)

        return out

    def poseAtention(self, x):

        # b = self.B(x)
        # c = self.C(x)
        # d = self.D(x)

        batch, ch, width, height = x.shape
        b =  x.reshape(batch, ch, width*height) # to [chanells, w*h]
        c =  x.reshape(batch, ch, width*height).mT # to [chanells, w*h].T
        d =  x.reshape(batch, ch, width*height)

        M_d = T.bmm( c,b)
        S = T.softmax(M_d, dim=1)
        out = T.matmul(d,S.mT).reshape(batch,ch, width, height)

        return out

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class Critic(nn.Module):
    def __init__(self, input_conv_dims, allias_state, enemy_state, lr = 0.001):
        super(Critic, self).__init__()

        self.checkpoint_file = 'tmp/Critic'
        self.input_dims = input_conv_dims
        self.allias_state = allias_state
        self.enemy_state = enemy_state

        self.conv1 = nn.Conv2d(in_channels=input_conv_dims[0], out_channels=8, kernel_size=8, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=4, stride=2, padding=0)
        self.pool = nn.AvgPool2d(kernel_size=4)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=0)
        self.norm = nn.BatchNorm2d(num_features = 32)
        self.flat = nn.Flatten(start_dim=1, end_dim= 3)

        self.alias_layer = CoordModel(allias_state, 8)
        self.enemy_layer = CoordModel(enemy_state, 8)

        conv_out = self.calc_conv_output()

        self.l1 = nn.Linear(conv_out + 8*2 , 128)
        self.l2 = nn.Linear(128, 64)
        self.l3 = nn.Linear(64, 32)
        self.V = nn.Linear(32, 1)

        self.optimizer = T.optim.RMSprop(self.parameters(), lr=lr)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def calc_conv_output(self):
        state = T.zeros(1, *self.input_dims)

        conv = self.conv1(state)
        conv = self.conv2(conv)
        conv = self.pool(conv)
        conv = self.conv3(conv)
        conv = self.norm(conv)
        conv = self.flat(conv)

        return conv.shape[1]

    def forward(self, state,E_coords, A_coords):

        conv = self.conv1(state)
        conv = self.conv2(conv)
        #############################################
        # chA   = self.chanelAtention(conv)
        # poseA = self.poseAtention(conv)
        # conv = T.add(chA, conv)
        # conv = T.add(poseA, conv)
        #############################################
        conv = self.pool(conv)
        conv = self.conv3(conv)
        #############################################
        # chA = self.chanelAtention(conv)
        # poseA = self.poseAtention(conv)
        # conv = T.add(chA, conv)
        # conv = T.add(poseA, conv)
        #############################################
        # conv = self.norm(conv)

        conv = self.flat(conv)
        a = self.alias_layer(A_coords)
        e = self.enemy_layer(E_coords)
        x = T.cat((conv, a, e), 1)

        x = F.elu(self.l1(x))
        x = F.elu(self.l2(x))
        x = F.elu(self.l3(x))
        value = self.V(x)

        return value

    def chanelAtention(self, x):

        batch, ch, width, height = x.shape

        b =  x.reshape(batch, ch, width*height) # to [chanells, w*h]
        c =  x.reshape(batch, ch, width*height).mT # to [chanells, w*h].T
        d =  x.reshape(batch, ch, width*height)

        M_d = T.matmul(b,c)
        S = T.softmax(M_d, dim=1)
        out = T.matmul(S.mT,d).reshape(batch,ch, width, height)

        return out

    def poseAtention(self, x):

        # b = self.B(x)
        # c = self.C(x)
        # d = self.D(x)

        batch, ch, width, height = x.shape
        b =  x.reshape(batch, ch, width*height) # to [chanells, w*h]
        c =  x.reshape(batch, ch, width*height).mT # to [chanells, w*h].T
        d =  x.reshape(batch, ch, width*height)

        M_d = T.bmm( c,b)
        S = T.softmax(M_d, dim=1)
        out = T.matmul(d,S.mT).reshape(batch,ch, width, height)

        return out

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


