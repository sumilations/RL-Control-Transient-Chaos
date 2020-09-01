import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D


class lorenzEnv(gym.Env):
	metadata = {'render.modes': ['human'],
	            'video.frames_per_second' : 30}
	           
	def __init__(self):
		self.n = 0
		self.t = 0
		self.rho   =  28.0
		self.sigma = 10.0
		self.beta  = 8.0/3.0
		self.tau   = 0.02
		high = np.array([self.rho/100.0, self.sigma/100.0, self.beta/100.0])
		high1 = np.array([np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max ])
		self.observation_space = spaces.Box(-high1, high1, dtype=np.float32)
		self.action_space = spaces.Box(-high, high, dtype=np.float32)
		self.fig = plt.figure()
		self.seed()
		self.viewer = None
		self.state = None
	 
	 
	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]
		        
	
	def step(self, action):
		
		#assert self.action_space.contains(action) #, "%r (%s) invalid"%(action, type(action))
		done=False
		sigmap, rhop, betap = action
		state=self.state
		
		def f(state, t):
			x, y, z, xdot, ydot, zdot = state	# unpack the state vectorstate=self.state
			#print(sigmap, action)
			return (self.sigma + sigmap) * (y - x), x * ((self.rho + rhop) - z) - y, x * y - (self.beta + betap) * z  # derivatives
			
		t=np.arange(self.tau, 3.0*(self.tau), self.tau)
		state2 = odeint(f, state, t)
		trofal = ((abs(state2[2,0]-state[0])<=5.) and (abs(state2[2,1]-state[1])<=5.) and (abs(state2[2,2]-state[2]))<=5.)
		
		if trofal:
			reward = 0
		else:
			reward = -1
		
		self.state = state2[1,:]
		
		return np.array(self.state), reward, done, {}
		
    	  
		
	def reset(self):
		self.state = self.np_random.uniform(low=-1, high=1, size=(6,))
		self.fig.clf() 
		#print(self.n)
		self.n=0
		return np.array(self.state) 
		
	def render(self, mode='human'):
		self.t +=self.tau
		self.n +=1
		plt.ion()
		ax = self.fig.gca(projection='3d')
		ax.scatter3D( self.state[0], self.state[1], self.state[2])# s=4, c='b')# )
		plt.show()
		if ((self.n)%(1000) == 0):
			plt.gcf().savefig('lorenz2.pdf')
		return np.array(self.state)
		
		
	def close(self):
		if self.viewer:
			self.viewer.close()
			self.viewer = None
			

    			
		    
		    
        
        	      
       
					
		
		
