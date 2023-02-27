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
    def __init__(self, input_dims, allias_state, enemy_state, n_actions,\
                 cuda, lr , gamma, max_actions, tau = 1.0):
        self.input_dim = input_dims[0]
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
                           max_actions = max_actions,
                           cuda = cuda,
                           lr = lr)
        self.critic = Critic(input_conv_dims = self.input_dim,
                             allias_state=allias_state,
                             enemy_state=enemy_state,
                             cuda=cuda,
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

        return action.cpu().numpy(), hx, cx

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

        next_v = T.zeros(1).to(self.actor.device) if done else \
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
    def __init__(self, input_conv_dims, cuda, allias_state, enemy_state,\
                 n_actions, max_actions, lr = 0.01):
        super(Actor, self).__init__()

        self.checkpoint_file = 'tmp/Actor'
        self.input_dims = input_conv_dims
        self.max_actions = max_actions
        self.allias_state = allias_state
        self.enemy_state = enemy_state

        self.obs1 = nn.Linear(in_features=input_conv_dims, out_features=64)
        self.obs2 = nn.Linear(in_features=64, out_features=32)
        self.obs3 = nn.Linear(in_features=32, out_features=4)

        self.flat = nn.Flatten(start_dim=0, end_dim=1)

        self.alias_layer = CoordModel(allias_state, 8)
        self.enemy_layer = CoordModel(enemy_state, 8)

        self.l1 = T.nn.LazyLinear(64)
        self.l2 = T.nn.Linear(64, 64)
        self.lstm = nn.LSTMCell(64,64)
        self.l3 = nn.Linear(64,32)

        self.mu = T.nn.Linear(32, n_actions)
        self.sigma = T.nn.Linear(32, n_actions)

        self.optimizer = T.optim.RMSprop(self.parameters(), lr=lr)
        self.device = T.device(cuda if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, E_coords, A_coords, hx, cx):

        x = self.obs1(state)
        x = T.relu(self.obs2(x))
        x = T.relu(self.obs3(x))
        x = self.flat(x)

        a = self.alias_layer(A_coords)
        e = self.enemy_layer(E_coords)

        x = T.cat((x, a, e), 0)

        x = F.tanh(self.l1(x))
        x = F.tanh(self.l2(x))
        hx, cx = self.lstm(x, (hx, cx))
        x = F.tanh(self.l3(hx))

        mu = 2 * T.tanh(self.mu(x))
        sigma = F.softplus(self.sigma(x)) + 0.001

        action_probs_d = T.distributions.Normal(mu, sigma)
        actions = action_probs_d.sample()

        log_probs = action_probs_d.log_prob(actions)
        log_probs = log_probs.sum(0, keepdim=True)

        return actions, log_probs, hx, cx


    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class Critic(nn.Module):
    def __init__(self, input_conv_dims, cuda,  allias_state, enemy_state, lr = 0.001):
        super(Critic, self).__init__()

        self.checkpoint_file = 'tmp/Critic'
        self.input_dims = input_conv_dims

        self.obs1 = nn.Linear(in_features=self.input_dims, out_features=64)
        self.obs2 = nn.Linear(in_features=64, out_features=32)
        self.obs3 = nn.Linear(in_features=32, out_features=4)

        self.alias_layer = CoordModel(allias_state, 8)
        self.enemy_layer = CoordModel(enemy_state, 8)

        self.flat = nn.Flatten(start_dim=0, end_dim=1)

        self.l1 = nn.LazyLinear(64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, 32)
        self.V = nn.Linear(32, 1)

        self.optimizer = T.optim.RMSprop(self.parameters(), lr=lr)
        self.device = T.device(cuda if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, E_coords, A_coords):

        x = self.obs1(state)
        x = T.relu(self.obs2(x))
        x = T.relu(self.obs3(x))

        x = self.flat(x)

        a = self.alias_layer(A_coords)
        e = self.enemy_layer(E_coords)

        x = T.cat((x, a, e), 0)

        x = F.tanh(self.l1(x))
        x = F.tanh(self.l2(x))
        x = F.tanh(self.l3(x))
        value = self.V(x)

        return value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))