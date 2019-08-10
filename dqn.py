import math
import random
import numpy as np
from collections import namedtuple

import torch.nn as nn
import torch.nn.functional as F


# DQN network model
#----------------------------------------------------------------------------------
class DQN(nn.Module):
	def __init__(self, actionNum):
		super(DQN, self).__init__()
		self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
		self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
		self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
		self.fc4 = nn.Linear(9 * 9 * 64, 512)
		self.fc5 = nn.Linear(512, actionNum)

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = F.relu(self.conv3(x))
		#print(x.size())
		x = F.relu(self.fc4(x.view(x.size(0), -1)))
		return self.fc5(x)
#----------------------------------------------------------------------------------


# Memory replay model
#----------------------------------------------------------------------------------
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class Memory(object):
	def __init__(self, capacity):
		self.capacity = capacity
		self.memory = []
		self.index = 0

	def remember(self, *args):
		if len(self.memory) < self.capacity:
			# Initializing the position
			self.memory.append(None)
		self.memory[self.index] = Transition(*args)
		self.index = (self.index + 1) % self.capacity

	def sample(self, batchSize):
		return random.sample(self.memory, batchSize)

	def getLength(self):
		return len(self.memory)
#----------------------------------------------------------------------------------


# Action selection and Optimization
# Should be moved to Game.py...
# ------------------------------------------------------------
# BATCH_SIZE = 50
# GAMMA = 0.999
# EPS_START = 0.9
# EPS_END = 0.01
# EPS_DECAY = 200
# TARGET_UPDATE = 100
#
# policy_dqn = DQN(3)
# target_dqn = DQN(3)
# target_dqn.load_state_dict(policy_dqn.state_dict())
# target_dqn.eval()
#
# steps_done = 0
# optim_done = 0
#
# memory = Memory(1000)
#
#
# def getAction(state):
# 	# Get epsilon
# 	global steps_done
# 	eps = EPS_END + (EPS_START - EPS_ENDS) * math.exp(-1. * steps_done/EPS_DECAY)
# 	steps_done += 1
# 	# Decide and return action as a tensor (use [item()] to get int when in "game.py")
# 	sample = random.random()
# 	if sample > eps:
# 		with torch.no_grad():
# 			return policy_dqn(state).max(1)[1].view(1,1)
# 	else:
# 		return torch.tensor([[random.randrange(n_actions)]], dtype=torch.long)
#
#
# def optimizeModel(memory):
# 	if len(memory) < BATCH_SIZE:
# 		return
#
# 	global optim_done
# 	# Sample and transpose memory data in batches
# 	transitions = memory.sample(BATCH_SIZE)
# 	batch = Transition(*zip(*transitions))
#
# 	state_batch = torch.cat(batch.state)
# 	action_batch = torch.cat(batch.action)
# 	reward_batch = torch.cat(batch.reward)
#
# 	# Generate the target values using [target_dqn]
# 	next_state_batch = torch.cat([s for s in batch.next_state
# 		if s is not None])
# 	next_state_mask = torch.tensor(tuple(map(lambda s: s is not None,
# 		batch.next_state)), dtype=torch.uint8)
#
# 	next_state_qValues = torch.zeros(BATCH_SIZE) # Customize max Q values in final state
# 	next_state_qValues[next_state_mask] = target_dqn(next_state_batch).max(1)[0].detach()
#
# 	target_values = (next_state_qValues * GAMMA) + reward_batch
#
# 	# Calculating the current values using [policy_dqn]
# 	current_values = policy_dqn(state_batch).gather(1, action_batch)
#
# # -----------------------------------------------------
# 	## May need to change this part using other loss functions...
# 	bellman_error = target_values - current_values
# 	d_error = -1. * bellman_error.clamp(-1, 1)
#
# 	optimizer.zero_grad()
# 	current_values.backward(d_error.data.unsqueeze(1))
# # -----------------------------------------------------
#
# 	optimizer.step()
#
# 	# Optimize target network occasionally
# 	if optim_done % TARGET_UPDATE == 0:
# 		target_dqn.load_state_dict(policy_dqn.state_dict())
#
# 	optim_done += 1
#
#
# # Main training body
# # Need to integrate with [game.py] somehow
# # -----------------------------------------------------
# EPISODES = 1000
#
# for episode_i in range(EPISODES):
#   # Using difference instead of [current_screen] to represent
#   # [state] of the game
#   current_screen = init_screen
#   current_state = current_screen - current_screen
#   done = False
#
#   while (not done):
#     action = getAction(state)
#     done, next_screen, reward = play(action)
#
#     if not done:
#       next_state = next_screen - current_screen
#     else:
#       next_state = None
#
#     memory.remember(current_state, action, next_state, reward)
#     optimizeModel(memory)
#
#     current_screen = next_screen
#     current_state = next_state
