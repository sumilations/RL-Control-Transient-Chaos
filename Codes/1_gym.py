import gym
import gym_lorenz
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, A2C
#from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN

env = gym.make('lorenz-v0')
#n_cpu = 2
#env = SubprocVecEnv([lambda: gym.make('lorenz-v0') for i in range(n_cpu)])
#env._max_episode_steps = 500
env = DummyVecEnv([lambda: env])  #The algorithms require a vectorized environment to run

model = PPO2(MlpLstmPolicy, env, nminibatches=1, verbose=1, tensorboard_log="./lorenztensorboard2/")
#model = PPO2('MlpLstmPolicy', 'lorenz-v0', nminibatches=1, verbose=1)
#model = PPO2.load('lorenz_targeting_Lstm_continous_2', env, nminibatches=1, verbose=1)
model.learn(total_timesteps=200000)
#model.save("lorenz_targeting_Lstm_continous_2")

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
