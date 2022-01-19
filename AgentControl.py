import torch
import numpy as np

from NNs import PolicyNN, CriticNN

class AgentControl:
    def __init__(self, input_state, output_action):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.action_shape = output_action
        self.moving_policy_nn = PolicyNN(input_state, output_action).to(self.device)
        self.target_policy_nn = PolicyNN(input_state, output_action).to(self.device)
        self.moving_critic_nn = CriticNN(input_state, output_action).to(self.device)
        self.target_critic_nn = CriticNN(input_state, output_action).to(self.device)
        self.noise_std = 0.1

    def get_action(self, state):
        actions = self.moving_policy_nn(torch.Tensor(state).to(self.device))
        noise = (self.noise_std**0.5)*torch.randn(self.action_shape).to(self.device) # ISPITATI TESKO ---------------------------------
        return np.clip((actions + noise).cpu().detach().numpy(), -1, 1)
