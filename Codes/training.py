import gym
import gym_lorenz
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
#from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN

env = gym.make('lorenz-v0')
#env._max_episode_steps = 500
env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run

model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log="./lorenztensorboard/")
model.learn(total_timesteps=500000)
model.save("lorenz_4")

#model = PPO2.load("lorenz_3")
#obs = env.reset()
#print(obs)
#print(env.action_space)
'''
for i in range(2000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
'''
