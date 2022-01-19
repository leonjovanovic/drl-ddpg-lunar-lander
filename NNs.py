from torch import nn
import torch

import Config

torch.manual_seed(Config.seed)
class PolicyNN(nn.Module):
    def __init__(self, input_state, output_action):
        super(PolicyNN, self).__init__()
        self.actions_means = nn.Sequential(
            nn.Linear(input_state, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, output_action)
        )
        self.actions_logstd = nn.Parameter(torch.zeros(output_action))

    def forward(self, state):
        actions_mean = self.actions_means(state)
        actions_std = torch.exp(self.actions_logstd)
        normal_distributions = torch.distributions.Normal(actions_mean, actions_std)
        actions = normal_distributions.sample()
        return actions

class CriticNN(nn.Module):
    def __init__(self, input_state, input_actions):
        super(CriticNN, self).__init__()
        self.value = nn.Sequential(
            nn.Linear(input_state + input_actions, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, state, actions):
        input_state_action = torch.cat(state, actions)
        return self.value(input_state_action)



