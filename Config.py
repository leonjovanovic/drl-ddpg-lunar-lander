import datetime

env_name = "LunarLanderContinuous-v2"

seed = 0
test_episodes = 100
test_every = 50000
number_of_steps = 1000000
buffer_size = 50000
min_buffer_size = 1000
batch_size = 64

# koliko dugo idemo random steps, treba nekih 10k msm
start_steps = 10000

hidden_sizes = [256, 256]
gamma = 0.99

decay = True
policy_lr = 0.0002
critic_lr = 0.0003

adam_eps = 1e-7
# Used to update target networks
polyak = 0.995
env_scale = True

writer_flag = True

decayy = "decay" if decay else ""
now = datetime.datetime.now()
date_time = "{}.{}.{}.{}".format(now.day, now.hour, now.minute, now.second)
writer_name = 'DDPG_LL-v2' + '_' + str(seed) + "_" + str(number_of_steps) + "_" + str(batch_size) + "_" + \
              str(start_steps) + "_" + str(gamma) + "_" + decayy + "_" + \
              str(policy_lr)[-2:] + "_" + str(critic_lr)[-2:] + "_" + \
              str(adam_eps)[-2:] + '_' + date_time
