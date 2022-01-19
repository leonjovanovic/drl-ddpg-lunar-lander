import Config
from AgentControl import AgentControl


class Agent:
    def __init__(self, input_state, output_action):
        self.agent_control = AgentControl(input_state, output_action)

    def get_action(self, state, n_step, env):
        if n_step < Config.start_steps:
            return env.action_space.sample()
        else:
            return self.agent_control.get_action(state)
