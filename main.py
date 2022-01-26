import gym
from tensorboardX import SummaryWriter
import Config
from Agent import Agent
from TestProcess import TestProcess

# --------------------------------------------------- Initialization ---------------------------------------------------
# Create Lunar Lander enviroment and add wrappers to record statistics
env = gym.make(Config.env_name)
env = gym.wrappers.RecordEpisodeStatistics(env)
state = env.reset()
# Create agent which will use DDPG to train NNs
agent = Agent(state.shape[0], env.action_space.shape[0])
# Initialize test process which will be occasionally called to test whether goal is met
test_process = TestProcess(state.shape[0], env.action_space.shape[0])
# Create writer for Tensorboard
writer = SummaryWriter(log_dir='content/runs/'+Config.writer_name) if Config.writer_flag else None
print(Config.writer_name)
# ------------------------------------------------------ Training ------------------------------------------------------
for n_step in range(Config.number_of_steps):
    # Check wether we should test the model
    if agent.check_test(test_process, n_step, writer, env):
        break
    #env.render()
    # Feed current state to the policy NN and get action
    actions = agent.get_action(state, n_step, env)
    # Use given action and retrieve new state, reward agent recieved and whether episode is finished flag
    new_state, reward, done, _ = env.step(actions)
    # Store step information to buffer for future use
    agent.add_to_buffer(state, actions, new_state, reward, done)
    # Update all 4 NNs
    agent.update(n_step)
    state = new_state
    if done:
        state = env.reset()
    # Print results to console and Tensorboard Writer
    agent.record_results(n_step, writer, env)
if writer is not None:
    writer.close()
test_process.env.close()
env.close()

#tensorboard --logdir="D:\Users\Leon Jovanovic\Documents\Computer Science\Reinforcement Learning\drl-ddpg-lunar-lander\content\runs" --host=127.0.0.1
