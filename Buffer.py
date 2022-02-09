import torch
import numpy as np


class Buffer:
    def __init__(self, state_size, action_size, buffer_capacity=100000):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.states = torch.zeros(buffer_capacity, state_size).to(self.device)
        self.actions = torch.zeros(buffer_capacity, action_size).to(self.device)
        self.new_states = torch.zeros(buffer_capacity, state_size).to(self.device)
        self.rewards = torch.zeros(buffer_capacity).to(self.device)
        self.dones = torch.zeros(buffer_capacity).to(self.device)
        self.buffer_counter = 0
        self.initialized = False
        self.buffer_size = buffer_capacity

    def add(self, state, actions, new_state, reward, done):

        self.states[self.buffer_counter] = torch.Tensor(state).to(self.device)
        self.actions[self.buffer_counter] = torch.Tensor(actions).to(self.device)
        self.new_states[self.buffer_counter] = torch.Tensor(new_state).to(self.device)
        self.rewards[self.buffer_counter] = torch.Tensor((reward,)).squeeze(-1).to(self.device)
        self.dones[self.buffer_counter] = torch.Tensor((1 if done else 0,)).squeeze(-1).to(self.device)

        self.buffer_counter = (self.buffer_counter + 1) % self.buffer_size
        if self.buffer_counter == 0 and not self.initialized:
            self.initialized = True

    def sample_indices(self, batch_size):
        indices = np.arange(min(self.buffer_counter, self.buffer_size) if not self.initialized else self.buffer_size)
        np.random.shuffle(indices)
        indices = indices[:batch_size]
        return indices
