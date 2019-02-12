# -*- coding: utf-8 -*-
"""
AI for a self driving car model

Created on Sun Feb 10 13:53:02 2019

@author: OzzyCodes
"""

"""
Deep Q-Learning Initialization Process:
    Va E A,s E S,Q0(a,s) = 1
"""

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F # Different functions used
import torch.optim as optim  # Optimizer
import torch.autograd as autograd
from torch.autograd import Variable


# Creating the architecture of the Neural Network
class Network(nn.Module): # Inherit all nn tool modules
    
    def __init__(self, input_size, nb_action): # Self is used to define the actual variable to use
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, 30)  # Specifies # of hidden layers
        self.fc2 = nn.Linear(30, nb_action)
        
    def forward(self, state):
        x = F.relu(self.fc1(state)) # Activated Hidden neurons
        q_values = self.fc2(x)
        return q_values


# Implementing Experience Replay
class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity # capacity of max transitions
        self.memory = []
    
    def push(self, transition):
        self.memory.append(transition) # Appends new event to last state
        if len(self.memory) > self.capacity:
            del self.memory[0] # delete first object
    
    def sample(self, batch_size):
        # if list = ((1,2,3), (4,5,6)), then zip(*list) --> ((1,4), (2,3), (5,6))
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)


# Implementing Deep Q Learning
class Dqn():

    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)  # Learning rate for AI of Adam class
        self.last_state = torch.Tensor(input_size).unsqueeze(0)  # New dimension of tensor
        self.last_action = 0
        self.last_reward = 0


    def select_action(self, state):
        probs = F.softmax(self.model(Variable(state, volatile = True)) * 100)  # Temperature = 100
        # softmax([1,2,3]) = [0.04,0.11,0.85] => softmax([1,2,3] * 3) = [0,0.02,0.98]
        action = probs.multinomial(1) # 1 is the number of sample to draw
        return action.data[0, 0]
    

    # Need to take the batches from the memory
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1) # Gathers the best action to play
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma * next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target) # td = temporal difference
        self.optimizer.zero_grad()
        td_loss.backward(retain_variables = True) # Free some memory when backtracking
        self.optimizer.step()


    # Update the last state and new reward along with all other elements after transition
    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_reward, batch_action = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action


    def score(self): # Sums all the elements in the reward_window
        return sum(self.reward_window)/(len(self.reward_window) + 1.) # Use plus 1 to avoid 0 which crashes program
    
    
    def save(self):  # Save your current module
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict,
                    }, 'last_brain.pth') # saved model in this path
        

    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint...")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict']) # Load the model
            self.optimizer.load_state_dict(checkpoint['optimizer'])  # Load the model parameters
            print("Done! ")
        else:
            print("No checkpoint found...")
