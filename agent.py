from collections import deque
import random
from nn import Dense, ReLU, Sequential
import numpy as np
from numpy import ndarray


MAX_MEMORY = 100_000
BATCH_SIZE = 1_000
LR = 0.001

class FFN:
    def __init__(self, input_size: int, hidden_size: int, action_size: int):
        self.network = Sequential([
            Dense(input_size, hidden_size),
            ReLU(),
            Dense(hidden_size, action_size)
        ], lr=LR)

    def forward(self, X: ndarray):
        self.network.forward(X)

class Trainer:
    def __init__(self, model, gamma):
        self.gamma = gamma
        self.model = model.network
        
    def train_step(self, state, action, reward, next_state, done):
        if len(state.shape) == 1:
            state = state[np.newaxis,:]
            next_state = next_state[np.newaxis, :]
            action = action[np.newaxis, :]
            reward = np.array([reward])[np.newaxis, :]
            done = (done,)
        
        pred: ndarray = self.model.forward(state)
        targets = np.copy(pred)
        
        for idx in range(len(done)):
            q_new = float(reward[idx])
            if not done[idx]:
                new_pred = self.model.predict(next_state[idx])
                q_new += self.gamma * np.max(new_pred)
            targets[idx,np.argmax(action[idx])] = q_new
        output_grad = self.model.loss.grad(targets,pred)
        self.model.backward(output_grad)
        
class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = FFN(15, 256, 3)
        self.trainer = Trainer(self.model, self.gamma)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def train_long_memory(self):
        sample = self.memory
        if len(self.memory) > BATCH_SIZE:
            sample = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*sample)
        self.trainer.train_step(np.array(states), np.array(actions), np.array(rewards), np.array(next_states), dones)
    
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
    
    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0,200) < self.epsilon:
            move = random.randint(0,2)
            final_move[move] = 1
        else:
            state0 = np.array(state,dtype=float)
            prediction = self.model.network.predict(state0)
            final_move[np.argmax(prediction)] = 1
        return np.array(final_move)
            

    
        


