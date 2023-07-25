import numpy 
from collections import deque, namedtuple
import numpy as np
from typing import Tuple
import random
import torch
import torch.nn as nn


BUFFER_SIZE = 1000
BATCH_SIZE = 16
EPSILON = 0.99
ACTION_SIZE = 4
STATE_SIZE = 37


class ReplayBuffer():
    def __init__(self,buffer_size:int, batch_size:int):
        self.buffer_size = buffer_size
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple('Experience', field_names= ['state','action','reward','next_state','done'])
        self.batch_size = batch_size
        

    def add(self, state: int, action: int, reward:int ,next_state:int , done:int ):
        self.memory.append(self.experience(state, action, reward, next_state, done))


    def sample(self) -> Tuple[ list, list, list , list]:
        experience = np.random.choice(a = self.memory, size = self.batch_size)

        state  = [ e['state'] for e in experience if e is not None]
        action = [ e['action'] for e in experience if e is not None]
        reward = [ e['reward'] for e in experience if e is not None]
        next_state = [ e['next_state'] for e in experience if e is not None]
        done = [ e['done'] for e in experience if e is not None]

        return (state, action, reward, next_state, done)
    
    def __len__(self):
        return len(self.memory)
    



class Agent():
    def __init__(self):
        self.replay_buffer = ReplayBuffer(buffer_size= BUFFER_SIZE, batch_size = BATCH_SIZE)
        self.epsilon = EPSILON
        self.action_size = ACTION_SIZE
        self.qnetwork = QNetwork(state_size=STATE_SIZE, action_size= ACTION_SIZE)
    

    def step(self):
        raise NotImplementedError()

    


    def act(self, state):
        if random.random() > self.epsilon:
            raise NotImplemented()
 
        else:
            return np.random.choice(range(self.action_size))

        


class QNetwork():
    def __init__(self,state_size:int, action_size:int):
        self.action_size = action_size,
        self.state_size = state_size

        self.network = nn.Sequential(
            nn.Linear((state_size, 64)),
            nn.ReLU(),
            nn.Linear((64, 128)),
            nn.Linear((128,256)),
            nn.ReLU(),
            nn.Linear( (256, self.action_size))
        )


    def forward(self, state):
        return self.network(state)
        


        

    
