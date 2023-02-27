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

        # self.batch_size = batch_size

    def generate_butches(self):

        return np.array(self.states), \
               np.array(self.actions), \
               np.array(self.probs), \
               np.array(self.vals), \
               np.array(self.rewards), \
               np.array(self.dones)
               # batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
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

class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha,
                 chkpt_dir='tmp/ppo'):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_ppo')

        self.l1 = nn.Linear(*input_dims, 64)
        self.r1 = nn.Tanh()
        self.l2 = nn.Linear(64, 64)
        self.r2 = nn.Tanh()
        self.GRU = nn.GRUCell(64, 32)

        self.mu = T.nn.Linear(32, n_actions)
        self.sigma = T.nn.Linear(32, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, h=None):
        x = self.r1(self.l1(state))
        x = self.r2(self.l2(x))
        h = self.GRU(x, h)

        mu = 2 * F.tanh(self.mu(h))
        sigma = F.softplus(self.sigma(h)) + 0.001

        dist = T.distributions.Normal(mu, sigma)

        return dist, h

    def forward_for_butch(self, state, h=None):
        x = self.r1(self.l1(state))
        x = self.r2(self.l2(x))
        h = self.GRU(x, h)

        mu = 2 * F.tanh(self.mu(h))
        sigma = F.softplus(self.sigma(h)) + 0.001

        return mu, sigma, h

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha,  chkpt_dir='tmp/ppo'):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_ppo')
        self.critic = nn.Sequential(
                nn.Linear(*input_dims, 256),
                nn.Tanh(),
                nn.Linear(256, 256),
                nn.Tanh(),
                nn.Linear(256,1),

        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self,state):
        value = self.critic(state)

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
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.003, gae_lambda=0.95,
                 policy_clip=0.2, n_epochs=10):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.n_epochs = n_epochs
        self.n_actions = n_actions

        self.actor = ActorNetwork(n_actions=n_actions, input_dims=input_dims, alpha=alpha)
        self.actor.apply(weights_init_uniform)

        self.critic = CriticNetwork(input_dims, alpha=alpha)
        # self.critic.apply(weights_init_uniform)

        self.memory = PPOMemory()

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def butch_predict(self, state_sec):
        h = None
        m = []
        s = []
        for i in state_sec:
            mu, sigma, h = self.actor.forward_for_butch(i, h)
            m.append(mu)
            s.append(sigma)
        dist = T.distributions.Normal(T.stack(m), T.stack(s))

        return dist

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation, hidden):
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)

        dist, hidden = self.actor(state, hidden)
        value = self.critic(state)
        action = dist.sample()

        probs = T.squeeze(dist.log_prob(action).sum(1, keepdim=True)).item()
        action = action.cpu().numpy()[0]
        value = T.squeeze(value).item()

        return action, probs, value, hidden

    def learn(self):
        for i in range(self.n_epochs):
            state_arr, action_arr, old_probs_arr, vals_arr, \
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
            old_probs = T.tensor(old_probs_arr).to(self.actor.device)
            actions = T.tensor(action_arr).to(self.actor.device)

            dist = self.butch_predict(states)
            critic_value = self.critic(states)

            critic_value = T.squeeze(critic_value)

            new_probs = dist.log_prob(actions).sum(1, keepdim=True)
            prob_ratio = new_probs.exp() / old_probs.exp()
            # prob_ratio = (new_probs - old_probs).exp()
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