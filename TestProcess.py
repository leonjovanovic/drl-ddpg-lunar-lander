import Config
from NNs import PolicyNN
import torch
import gym
import numpy as np
import json

class TestProcess:
    def __init__(self, input_state, output_action):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.policy_nn = PolicyNN(input_state, output_action).to(self.device)
        self.env = None

    def test(self, writer, trained_policy, env_episode):
        self.policy_nn.load_state_dict(trained_policy.state_dict())
        self.env = gym.make(Config.env_name)
        self.env = gym.wrappers.RecordEpisodeStatistics(self.env)
        # Create new enviroment and test it for 100 episodes using the model we trained
        print("Testing...")
        print("[", end="")
        state = self.env.reset()
        for n_episode in range(Config.test_episodes):
            while True:
                actions = self.policy_nn(torch.Tensor(state).to(self.device))
                new_state, reward, done, _ = self.env.step(actions.cpu().detach().numpy())
                state = new_state
                if done:
                    state = self.env.reset()
                    print('.', end="")
                    break
        mean_return = np.mean(self.env.return_queue)
        print("]")
        print(self.env.return_queue)
        # Add results to TensorBoard
        if writer is not None:
            writer.add_scalar('testing 100 reward', mean_return, env_episode)
        return self.check_goal(mean_return)

    def check_goal(self, mean_return):
        if mean_return < 200:
            print("Goal NOT reached! Mean 100 test reward: " + str(np.round(mean_return, 2)))
            return False
        else:
            print("GOAL REACHED! Mean reward over 100 episodes is " + str(np.round(mean_return, 2)))
            # If we reached goal, save the model locally
            torch.save(self.policy_nn.state_dict(), 'models/model' + Config.date_time + '.p')
            #self.record_final_episode()
            return True

    def record_final_episode(self):
        self.env = gym.wrappers.RecordVideo(self.env, "bestRecordings", name_prefix="rl-video" + Config.date_time, )
        state = self.env.reset()
        while True:
            actions = self.policy_nn(torch.Tensor(state).to(self.device))
            new_state, reward, done, _ = self.env.step(actions.cpu().detach().numpy())
            state = new_state
            if done:
                break
