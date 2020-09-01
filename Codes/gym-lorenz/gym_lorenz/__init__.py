from gym.envs.registration import register

register(
    id='lorenz-v0',
    entry_point='gym_lorenz.envs:lorenzEnv',
    max_episode_steps = 400000,
    reward_threshold  = 1000
)
