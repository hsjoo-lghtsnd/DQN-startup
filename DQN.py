# pytorch document provides warning for not to use python built-in 'random' package
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class MyDeepQNet(nn.Module):
    def __init__(self, num_states, num_actions):
        print("Agent Q-Network is initiated.")
        super(MyDeepQNet, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.linear1 = nn.Linear(num_states, 128).to(device)
        self.linear2 = nn.Linear(128, 256).to(device)
        self.linear3 = nn.Linear(256, 64).to(device)
        self.linear4 = nn.Linear(64, num_actions).to(device)

    def forward(self, x):
        # the model returns Q(s,a) function itself.
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        return self.linear4(x)
    
    def get_dtype(self):
        return self.linear1.weight.dtype


class environment:
    def __init__(self, MAX_VALUE):
        print("a simple environment is initiated.")

        self.state = MAX_VALUE/2.       # initial: center
        self.momentum = 0.              # initial: zero speed
        self.MAX = MAX_VALUE

    def action(self, choice):
        self.momentum = self.momentum + 0.2*(choice - self.state)
        self.state = self.state + 0.1*momentum

        reward = (choice - self.state)**2
        if ((self.state > self.MAX) or (self.state < 0)):
            reward = -100.

        return reward

    def get_state(self): # observes current position (state)
        return self.state
    
    def get_momentum(self): # observes current momentum (additional state) with noise
        noise = torch.randint(-50,50+1,(1,))
        n = noise.numpy()/10.
        return self.momentum + n

    def get_momentum_clear(self): # observes current momentum without noise
        return self.momentum
        

class DQNAgent:
    def __init__(self, env, MAX_VALUE, MOMENTUM_SIZE, GAMMA, _lr_=1e-5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        self.model = MyDeepQNet(MOMENTUM_SIZE+1, MAX_VALUE) # depends on environment!!
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=_lr_)
    
        self.env = env

        self.MSIZE = MOMENTUM_SIZE  # for state construction.
        self.MAX = MAX_VALUE        # for random action. depends on environment.
        self.GAMMA = GAMMA
    
        # these are reserved for debug.
        self._states = []
        self._rewards = []
        self._losses = []

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def get_gamma(self):
        return self.GAMMA

    def _put_state(self, s0, s1):
        self._states.append((s0, s1))
        return

    def _get_state(self):
        return self._states

    def _put_reward(self, r):
        self._rewards.append(r)
        return

    def _get_reward(self):
        return self._rewards

    def _put_loss(self, l):
        self._losses.append(l)
        return

    def _get_loss(self):
        return self._losses

    def observe(self): # env-dependent!!
        state = []
        state.append(self.env.get_state())

        for i in range(self.MSIZE):
            state.append(self.env.get_momentum())
        return state

    def random_choice(self):
        return torch.randint(0,self.MAX+1,(1,)).numpy()

    def get_Q(self, state):
        s = torch.tensor(state, device = self.device, dtype = self.model.get_dtype())
        return self.model.forward(s)

    def Q_choice(self, state):
        Q = self.get_Q(state)
        action = np.argmax(Q)
        return action

def train(agent, iter_num, epsilon, __DEBUG__=False, __SHOW_ITER__=500):
    g = agent.get_gamma()
    for i in range(iter_num):
        # sample data.
        s0 = agent.observe()
        if (torch.rand(1) > epsilon):
            a = agent.Q_choice(s0)
        else:
            a = agent.random_choice()
        s1 = agent.observe()
        r = agent.env.action(a)

        agent.optimizer.zero_grad()

        Q0 = agent.get_Q(s0)
        Q1 = agent.get_Q(s1)
        y = r + (g*max(Q))
        loss = (y - Q0)**2
        
        loss.backward()
        agent.optimizer.step()

        if (__DEBUG__):
            agent._put_state(s0, s1)
            agent._put_reward(r)
            agent._put_loss(loss)

        if (i%__SHOW_ITER__ == 0):
            print("Iteration:", t, ",\t", "loss = ", loss.item())

