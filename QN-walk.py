from TrainMLP import *
import random

ENV = ['X', ' ', ' ', ' ', ' ', ' ', 'O']
ACTIONS = [-1, 1]

ALPHA = 0.1
GAMMA = 0.9

Q = { 	0: [0, 0],  
		1: [0, 0], 
		2: [0, 0], 
		3: [0, 0], 
		4: [0, 0], 
		5: [0, 0], 
		6: [0, 0], }

def Reward(state, action):
	state_p = state + action
	if ENV[state_p] == 'X':
		return -1
	elif ENV[state_p] == 'O':
		return 1
	else:
		return 0

def StateTrans(state, action_index):
	action = ACTIONS[action_index]
	state_p = state + action
	r = Reward(state, action)
	return state_p, r

def IsTerminal(state):
	return True if ENV[state] != ' ' else False
	
def DetermineActionIndex(state, epsilon):
	p = random.uniform(0, 1)
	if p < epsilon:
		index = random.randint(0, 1)
	else:
		qlist = Q[state]
		index = qlist.index(max(qlist)) 
	return index

def MaxQ(state):
	qlist = Q[state]
	if IsTerminal(state) == True:
		return 0
	else:
		return max(qlist)
	

def UpdateQ(state, action_index, reward, state_p):
	qlist = Q[state]
	q_old = qlist[action_index]
	q_new = q_old + ALPHA*(reward + GAMMA*MaxQ(state_p) - q_old)
	qlist[action_index] = q_new
	Q[state] = qlist

epsilon = 0.9
for episode in range(1000):
	state = 3
	steps = 0
	while IsTerminal(state) == False:
		action_index = DetermineActionIndex(state, epsilon)
		state_p, r = StateTrans(state, action_index)
		UpdateQ(state, action_index, r, state_p)
		steps += 1
		state = state_p

	# anneal epsilon 50 times in training
	if epsilon > 0 and episode %(1000//50) == 0:
		epsilon *= 0.9
	print(episode,': steps = %d, result = %s'%(steps, ENV[state]))
# Output Q table
for state in range(len(ENV)):
	print(state, '=> %3.2f,	%3.2f'%(Q[state][0], Q[state][1]))



