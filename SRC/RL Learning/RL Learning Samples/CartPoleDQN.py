# First Attempt at Learning, Trying a different tutorial
import gymnasium as gym
import random
import math
from itertools import count


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 

from collections import deque, namedtuple


device = torch.device("cuda")
learning_rate = 0.001
Epsilon = 0.05 # 5% chance to choose a random action
env = gym.make("CartPole-v1") #, render_mode='human')
states = env.observation_space.shape[0]
actions = env.action_space.n

# Random games for sanity checks
# def RandomGames():
#     RGames = 5             # 10 Games
#     GameDuration = 500      # 500 time lengths per game
#     for Game in range(RGames):
#         env.reset()
        
#         for t in range(GameDuration):
#             #env.render()
                        
#             actionRand = env.action_space.sample() # Selects one of the possible action states
            
#             next_state, reward, done, truncation, info = env.step(actionRand)
            
#             print(t, next_state, reward, done, truncation, info, actionRand)
#             if done:
#                 break            
# RandomGames()

# Network that defines our policy and target networks
class CartPoleNN(nn.Module):
    def __init__(self, observation_space, action_space):
        super(CartPoleNN, self).__init__()
        self.fc1 = nn.Linear(observation_space, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_space)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
Transition = namedtuple('Transition',('state','action','next_state','reward', 'terminated'))

# Class that represent our memory
class ReplayMemory(object):
    def __init__ (self, capacity): # On Initialization
        self.memory = deque([], maxlen = capacity) # Creates a que with a set maximum size
        
    def push(self,*args): # When we want to push an undefined size of variables in *args to the system
        self.memory.append(Transition(*args))
        
    def sample(self, batch_size): # Returns a random set of memory elements
        return random.sample(self.memory, batch_size)
    
    def __len__(self): # When we want to know the length of the memory thus far
        return len(self.memory)


# Model Info
output_size = env.action_space.n
input_size = int(env.observation_space.shape[0])

batch_size = 64
gamma = 0.99
Eps_Start = 0.9 # Start with random decisions 90% of the time
Eps_End = 0.05 # random decision choice shrinks to 5% as time goes on
Eps_Decay = 1000 # Defines rate at which we go from epsilon start to end
learning_rate = 0.001
Tau = 0.005

device = torch.device('cpu')

# Initialize Network
PolicyModel = CartPoleNN(observation_space = input_size, action_space = output_size).to(device)
PolicyDict = PolicyModel.state_dict()

# Start by Copying PolicyModel on Initialization
TargetModel = CartPoleNN(observation_space = input_size, action_space = output_size).to(device)
TargetDict = PolicyDict
TargetModel.load_state_dict(TargetDict)

# Loss and Optimizer
criterion = nn.SmoothL1Loss() # Choosing between multiple labels
optimizer = optim.Adam(PolicyModel.parameters(), lr= learning_rate)

memory = ReplayMemory(100000) # Creates memory object that allows for 10k game sessions

steps_done = 0
def ActionSelect(state):
    global steps_done # pulls the steps_done variable
    EpsilonCheck = random.random()
    
    eps_threshold = Eps_End + (Eps_Start-Eps_End) * math.exp(-1 * (steps_done/Eps_Decay)) # Random Chance value
    #eps_threshold = 0 # For testing purposes
    if EpsilonCheck > eps_threshold:
        # Do Policy Based Decision
        ActionLabels = PolicyModel(state)
        action = int(torch.argmax(ActionLabels)) # label with highest likelihood
    else:
        # Do Random Action
        action = env.action_space.sample()
    steps_done += 1 # add 1 to steps_done variable
    return action


Init_State = env.reset()
Init_State = torch.FloatTensor(torch.from_numpy(Init_State[0]))
Action = ActionSelect(Init_State)

# The Optimization Function
def optimize_model():
    if len(memory) < batch_size: # returns nothing if memory is too small to do training
        return
    transitions = memory.sample(batch_size) # samples batch_size entries from memory
    batch = Transition(*zip(*transitions)) # 
    
    state_batch = torch.stack(batch.state,) 
    reward_batch = torch.stack(batch.reward)
    action_batch = torch.tensor(batch.action)
    action_batch = action_batch.reshape(reward_batch.size())
    terminated_batch = torch.tensor(batch.terminated)
    terminated_batch = terminated_batch.reshape(reward_batch.size())
    # Values we would have used given the PolicyModel
    current_q_list  = PolicyModel(state_batch)
    with torch.no_grad():
        target_q_list = TargetModel(state_batch)
    # Initialize next_state_values
    for index ,TermFlag in enumerate(terminated_batch):
        if TermFlag:
            print("Terminated")
            target = reward_batch[index]
        else:
            print("Not Terminal")
            target = reward_batch[index] + gamma*target_q_list[index].max()
        target_q_list[index] = target
    
    criterion = nn.SmoothL1Loss()
    loss = criterion(current_q_list, target_q_list)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(PolicyModel.parameters(),100)
    optimizer.step()
    print("Checkpoint")
    
update_Rate = 10
    
num_episodes = 50
EpisodeLen = 500

# Somethine We'll plot later
Episode_Durations = []

for i_episode in range(num_episodes):
    state, info =  env.reset()
    state = torch.FloatTensor(torch.from_numpy(state))
    
    for t in count():
        action = ActionSelect(state)
        observe, reward, terminated, truncation, info = env.step(action)
        reward = torch.tensor([reward], device = device)
        done = terminated or truncation
        
        if truncation: # If stat action leads to end
            next_state = None#
        
        else: # If State action doesn't end
            next_state = torch.tensor(observe, dtype = torch.float32, device = device).unsqueeze(0) # put next state observation into the next_state, unsqueeze just reshapes to desired size
        
        memory.push(state, action, next_state, reward, terminated)
        
        # Optimize PolicyDict
        optimize_model()
        
        PolicyDict = PolicyModel.state_dict()
        TargetDict = TargetModel.state_dict()
        
        for key in PolicyDict: # After PolicyDict is updated, for each value, update TargetDict
            TargetDict[key] = PolicyDict[key]*Tau + TargetDict[key]*(1-Tau)
        TargetModel.load_state_dict(TargetDict)

        if done:
            break
        
print("Complete")














# def Random_or_PolicyGames(epsilon):
#     RGames = 20             # 10 Games
#     GameDuration = 500      # 500 time lengths per game
#     for Game in range(RGames):
#         ResetState = env.reset() # state is a tuple when leaving this value
#         state = ResetState[0] # Pull State Vector from tuple after reset
#         for t in range(GameDuration):
#             #env.render()
#             Policymodel.eval()
                        
#             Epsilon_Check = random.random()          
#             # Choose to use policy or random sample
#             if Epsilon_Check < Epsilon:
#                 action = env.action_space.sample() # Selects one of the possible action states
#                 print("Rand Action")
#             else:
#                 print("Model Action")
#                 with torch.no_grad():
#                     state = torch.from_numpy(state)
#                     state = torch.FloatTensor(state)
#                     ActionLabels = Policymodel(state)
#                     action = int(torch.argmax(ActionLabels))
                    
#             next_state, reward, done, truncation, info = env.step(action) # State is numpy array when leaving this area
#             print(t, next_state, reward, done, truncation, info, action)
            
#             state = next_state # 
#             if done:
#                 break    
# Random_or_PolicyGames(.3)



