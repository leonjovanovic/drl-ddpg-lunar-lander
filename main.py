import gym
import Config
from Agent import Agent

env = gym.make("LunarLanderContinuous-v2")
state = env.reset()
print(state.shape[0])
print(env.action_space.shape[0])
agent = Agent(state.shape[0], env.action_space.shape[0])

for n_step in range(Config.number_of_steps):
    #env.render()
    actions = agent.get_action(state, n_step, env)
    print(actions)
    new_state, reward, done, _ = env.step(actions)
    state = new_state
    if done:
        state = env.reset()
