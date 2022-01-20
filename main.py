import gym
from tensorboardX import SummaryWriter
import Config
from Agent import Agent
from TestProcess import TestProcess

env = gym.make(Config.env_name)
env = gym.wrappers.RecordEpisodeStatistics(env)
state = env.reset()
agent = Agent(state.shape[0], env.action_space.shape[0])
test_process = TestProcess(state.shape[0], env.action_space.shape[0])
writer = SummaryWriter(log_dir='content/runs/'+Config.writer_name) if Config.writer_flag else None

for n_step in range(Config.number_of_steps):
    if agent.check_test(test_process, n_step, writer, env):
        break
    #env.render()
    actions = agent.get_action(state, n_step, env)
    new_state, reward, done, _ = env.step(actions)
    agent.add_to_buffer(state, actions, new_state, reward, done)
    agent.update(n_step)
    state = new_state
    if done:
        state = env.reset()
    agent.record_results(n_step, writer, env)

writer.close()
test_process.env.close()
env.close()

#NORMALIZOVATI ENV
# SREDITI NOISE
# ANNEAL NOISA I LR