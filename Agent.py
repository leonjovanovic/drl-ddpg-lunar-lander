import itertools

import Config
from AgentControl import AgentControl
from Buffer import Buffer
from collections import deque
import numpy as np


class Agent:
    def __init__(self, state_size, action_size):
        self.agent_control = AgentControl(state_size, action_size)
        self.buffer = Buffer(state_size, action_size, buffer_capacity=Config.buffer_size)
        self.critic_loss_mean = deque(maxlen=100)
        self.policy_loss_mean = deque(maxlen=100)
        self.max_reward = -300
        self.ep_count = -1

    def get_action(self, state, n_step, env):
        if n_step < Config.start_steps:
            return env.action_space.sample()
        else:
            return self.agent_control.get_action(state)

    def add_to_buffer(self, state, actions, new_state, reward, done):
        self.buffer.add(state, actions, new_state, reward, done)

    def update(self, n_step):
        self.agent_control.lr_decay(n_step)
        if self.buffer.buffer_counter < Config.min_buffer_size:
            return
        indices = self.buffer.sample_indices(Config.batch_size)
        critic_loss = self.agent_control.update_critic(self.buffer.states[indices], self.buffer.actions[indices],
                                                       self.buffer.rewards[indices], self.buffer.new_states[indices],
                                                       self.buffer.dones[indices])
        policy_loss = self.agent_control.update_policy(self.buffer.states[indices])
        self.agent_control.update_targets()
        self.critic_loss_mean.append(critic_loss)
        self.policy_loss_mean.append(policy_loss)

    def record_results(self, n_step, writer, env):
        if self.buffer.buffer_counter < Config.min_buffer_size or self.ep_count == env.episode_count:
            return
        self.ep_count = env.episode_count
        self.max_reward = np.maximum(self.max_reward, np.max(env.return_queue))
        print("Ep " + str(self.ep_count) + "/" + str(Config.number_of_steps) + " St " + str(
            n_step) + " Mean 100 policy loss: " + str(
            np.round(np.mean(self.policy_loss_mean), 4)) + " Mean 100 critic loss: " + str(
            np.round(np.mean(self.critic_loss_mean), 4)) + " Max reward: " + str(
            np.round(self.max_reward, 2)) + " Mean 100 reward: " + str(
            np.round(np.mean(env.return_queue), 2)) + " Last rewards: " + str(
            np.round(env.return_queue[-1], 2)))

        if Config.writer_flag:
            writer.add_scalar('pg_loss', np.mean(self.policy_loss_mean), self.ep_count)
            writer.add_scalar('vl_loss', np.mean(self.critic_loss_mean), self.ep_count)
            writer.add_scalar('100rew', np.mean(env.return_queue), self.ep_count)
            writer.add_scalar('rew', env.return_queue[-1], self.ep_count)

    def check_test(self, test_p, n_step, writer, env):
        if (n_step + 1) % Config.test_every == 0 or (
                len(env.return_queue) >= 100 and np.mean(list(itertools.islice(env.return_queue, 60, 100))) >= 200):
            return test_p.test(writer, self.agent_control.moving_policy_nn, env.episode_count)
        return False
