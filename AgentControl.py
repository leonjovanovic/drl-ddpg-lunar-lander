import torch
import numpy as np

import Config
from NNs import PolicyNN, CriticNN


class AgentControl:
    def __init__(self, input_state, output_action):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.action_shape = output_action
        self.moving_policy_nn = PolicyNN(input_state, output_action).to(self.device)
        self.policy_nn_optim = torch.optim.Adam(params=self.moving_policy_nn.parameters(), lr=Config.policy_lr,
                                                eps=Config.adam_eps)
        self.moving_critic_nn = CriticNN(input_state, output_action).to(self.device)
        self.critic_nn_optim = torch.optim.Adam(params=self.moving_critic_nn.parameters(), lr=Config.critic_lr,
                                                eps=Config.adam_eps)
        self.target_policy_nn = PolicyNN(input_state, output_action).to(self.device)
        self.target_policy_nn.load_state_dict(self.moving_policy_nn.state_dict())
        self.target_critic_nn = CriticNN(input_state, output_action).to(self.device)
        self.target_critic_nn.load_state_dict(self.moving_critic_nn.state_dict())
        self.mse = torch.nn.MSELoss()

        self.noise_std = 0.1

    def get_action(self, state):
        actions = self.moving_policy_nn(torch.Tensor(state).to(self.device))
        noise = (self.noise_std ** 0.5) * torch.randn(self.action_shape).to(
            self.device)  # ISPITATI TESKO ---------------------------------
        return np.clip((actions + noise).cpu().detach().numpy(), -1, 1)

    def lr_decay(self, n_step):
        if Config.decay:
            frac = 1 - n_step/Config.number_of_steps
            self.policy_nn_optim.param_groups[0]["lr"] = frac * Config.policy_lr
            self.critic_nn_optim.param_groups[0]["lr"] = frac * Config.critic_lr


    def update_critic(self, states, actions, rewards, new_states, dones):
        new_actions = self.target_policy_nn(new_states).detach()
        target_values = self.target_critic_nn(new_states, new_actions).squeeze(-1).detach()
        target = rewards + Config.gamma * target_values * (1 - dones)
        state_values = self.moving_critic_nn(states, actions).squeeze(-1)
        critic_loss = self.mse(state_values, target)

        self.critic_nn_optim.zero_grad()
        critic_loss.backward()
        self.critic_nn_optim.step()

        return critic_loss.cpu().detach().numpy()

    def update_policy(self, states):
        policy_actions = self.moving_policy_nn(states)
        critic_value = self.moving_critic_nn(states, policy_actions).squeeze(-1)
        # Used `-value` as we want to maximize the value given
        # by the critic for our actions
        policy_loss = -torch.mean(critic_value)

        self.policy_nn_optim.zero_grad()
        policy_loss.backward()
        self.policy_nn_optim.step()

        return policy_loss.cpu().detach().numpy()

    def update_targets(self):
        # update target networks by polyak averaging.
        with torch.no_grad():
            for mov, targ in zip(self.moving_critic_nn.parameters(), self.target_critic_nn.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                targ.data.mul_(Config.polyak)
                targ.data.add_((1 - Config.polyak) * mov.data)

            for mov, targ in zip(self.moving_policy_nn.parameters(), self.target_policy_nn.parameters()):
                targ.data.mul_(Config.polyak)
                targ.data.add_((1 - Config.polyak) * mov.data)
