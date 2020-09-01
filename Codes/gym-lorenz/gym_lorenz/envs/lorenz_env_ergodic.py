import gym
import csv
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
		self.tau   = 0.002
		self.x_0 = 0
		self.y_0 = 0
		self.z_0 = 0
		high = np.array([self.rho, self.sigma, self.beta])
		self.space_a = np.array([-self.rho/10.0, -self.rho/100.0, 0, self.rho/100.0, self.rho/80.0])
		self.space_c = np.array([-self.beta/80.0, -self.beta/100.0, 0, self.beta/100.0, self.beta/80.0])
		self.space_b = np.array([-self.sigma/80.0, -self.sigma/100.0, 0, self.sigma/100.0, self.sigma/80.0])
		high1 = np.array([np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max  ])
		self.observation_space = spaces.Box(-high1, high1, dtype=np.float32)
		self.action_space = spaces.Box(-high, high, dtype=np.float32)
		#self.action_space = spaces.MultiDiscrete([5, 5, 5])
		self.fig = plt.figure()
		self.seed()
		self.viewer = None
		self.state = None
	 
	 
	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]
		        
	
	def step(self, action):
		
		assert self.action_space.contains(action) , "%r (%s) invalid"%(action, type(action))
		done=False
		
		rhop, sigmap, betap= action
		state=self.state[0:3]
		
		def f(state, t):
			x, y, z = state	# unpack the state vectorstate=self.state
			#print(sigmap, action)
			return (self.sigma + sigmap) * (y - x), x * ((self.rho + rhop) - z) - y, x * y - (self.beta + betap) * z  # derivatives when continous action space
			#return (self.sigma + self.space_b[sigmap]) * (y - x), x * ((self.rho + self.space_a[rhop]) - z) - y, x * y - (self.beta + self.space_c[betap]) * z  #derivatives when discrete action space
			
		t=np.arange(self.tau, 4.0*(self.tau), self.tau)
		state2 = odeint(f, state, t)
		xdot, ydot, zdot = f(state2[2,:], t)
		
		#trofal = ((abs(state2[2,0]-state[0])<1.) and (abs(state2[2,1]-state[1])<1.) and (abs(state2[2,2]-state[2]))<1.)
		r = np.sqrt(((state2[2,0] - self.x_0)**2) + ((state2[2,1]- self.y_0)**2) + ((state2[2,2] - self.z_0))**2)/65.
		#r = np.sqrt(((state2[2,0] - state[0])**2) + ((state2[100,1]- state[1])**2) + ((state2[100,2] - state[2]))**2)/65.
		
		
		if r>0.1:
			reward = -r
		
		else:
			reward = (r+0.0001)**(-1)#np.exp(-r) #-np.exp(-np.exp(-r))
				
		
		
		
		#reward = -(((state2[2,0]-state[0])**2 ) +((state2[2,1]-state[1])**2) +((state2[2,2]-state[2]))**2) - 1.0)
		
		'''
		if trofal:
			reward = 1
		else:
			reward = -1
		
		'''
		self.state = state2[2,0], state2[2,1], state2[2,2], xdot, ydot, zdot 
		
		#row = [self.t, self.state[0], self.state[1], self.state[2], self.space_a[rhop], self.space_b[sigmap], self.space_c[betap]] # for discrete action space
		row = [self.t, self.state[0], self.state[1], self.state[2], rhop, sigmap, betap, r, reward] #for continous action space
		self.t +=self.tau	
		
		with open('lorenz_9.csv', 'a') as output:
			writer = csv.writer(output)
			writer.writerow(row)
			
		output.close()
			
		return np.array(self.state), reward, done, {}
		
    	  
		
	def reset(self):
		self.state = self.np_random.uniform(low=-40, high=50, size=(6,))
		self.fig.clf() 
		#print(self.n)
		self.n=0
		return np.array(self.state) 
		
	def render(self, mode='human'):
		
		self.n +=1
		#plt.ion()
		#ax = self.fig.gca(projection='3d')
		#ax.scatter3D( self.state[0], self.state[1], self.state[2])# s=4, c='b')# )
		#plt.show()
		#if ((self.n)%(1000) == 0):
		#	plt.gcf().savefig('lorenz2.pdf')
		#return np.array(self.state)
		
		
	def close(self):
		if self.viewer:
			self.viewer.close()
			self.viewer = None
			

    			
		    
		    
        
        	      
       
					
		
		
