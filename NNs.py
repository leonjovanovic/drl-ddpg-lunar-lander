from torch import nn
import torch

import Config

torch.manual_seed(Config.seed)
class PolicyNN(nn.Module):
    def __init__(self, input_state, output_action):
        super(PolicyNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_state, Config.hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(Config.hidden_sizes[0], Config.hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(Config.hidden_sizes[1], output_action),
            nn.Tanh()
        )

    def forward(self, state):
        actions = self.model(state)
        return actions

class CriticNN(nn.Module):
    def __init__(self, input_state, input_actions):
        super(CriticNN, self).__init__()
        self.value = nn.Sequential(
            nn.Linear(input_state + input_actions, Config.hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(Config.hidden_sizes[0], Config.hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(Config.hidden_sizes[0], 1)
        )

    def forward(self, state, actions):
        input_state_action = torch.cat((state, actions), 1)
        return self.value(input_state_action)



