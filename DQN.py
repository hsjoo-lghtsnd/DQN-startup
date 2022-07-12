import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.utils.epsilon_greedy import Epsilon
from src.utils.general_functions import torch_from_frame
from src.utils.replay_memory import Transition, ReplayMemory

class MyNet(nn.Module):
  def __init__(self, num_states, num_actions):
    super(MyNet, self).__init__()

    self.linear1 = nn.Linear(num_states, 128)
    self.linear2 = nn.Linear(128, 256)
    self.linear3 = nn.Linear(256, 64)
    self.linear4 = nn.Linear(64, num_actions)

  def forward(self, x):
    x = F.relu(self.linear1(x))
    x = F.relu(self.linear2(x))
    x = F.relu(self.linear3(x))
    return self.linear4(x)

class environment:
  def __init__(self):
    
  def action
class DQNAgent:
  def __init__(self, dnn, env, memory):
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    self.model = dnn
    self.optimizer = optim.RMSprop(self.model.parameters(), lr=1e-5)
    
    self.env = env
    
    # these are for debug.
    self._rewards = []
    self._losses = []
    
  
