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
		self.n = 2
		self.t = 0
		self.rho   =  28.0
		self.sigma = 10.0
		self.beta  = 8.0/3.0
		self.tau   = 0.002
		self.x_0 = 2
		self.y_0 = 3
		self.z_0 = 4
		self.o_0 = 0
		self.o_1 = 0
		self.o_2 = 0
		#high = np.array([self.rho, self.sigma, self.beta])
		#high = np.array([self.rho])
		high = np.array([np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max  ])
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
		self.reward = None
		self.max_reward = 0
		self.epsilon = 0.001
		#self.n = 1
	 
	 
	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]
		        
	
	def step(self, action):
		
		assert self.action_space.contains(action) , "%r (%s) invalid"%(action, type(action))
		done=False
		
		rhop, sigmap, betap = action
		#rhop = action
		state=self.state[0:3]
		tof = np.absolute(self.state[0:3]) < self.epsilon*np.ones(3)
		
		if ((tof.all()) and (self.reward!=self.max_reward)) :
			state = self.state[0:3] + 10*np.random.randn(3)
		
		def f(state, t):
			x, y, z = state	# unpack the state vectorstate=self.state

			return (self.sigma + sigmap) * (y - x), x * ((self.rho + rhop) - z) - y, x * y - (self.beta + betap) * z  # derivatives when continous action space
			#return (self.sigma + self.space_b[sigmap]) * (y - x), x * ((self.rho + self.space_a[rhop]) - z) - y, x * y - (self.beta + self.space_c[betap]) * z  #derivatives when discrete action space
			#return (self.sigma + sigmap) * (y - x), x * ((self.rho + rhop) - z) - y, x * y - (self.beta) * z 
			
			
		t=np.arange(self.t+self.tau,  self.t + (self.n + 2)*(self.tau), self.tau)
		state2 = odeint(f, state, t)
		xdot, ydot, zdot = f(state2[self.n,:], t)
		x_0 = np.sin((2.*np.pi)*t)
		state3 = state2[:,1]/(np.sqrt(np.sum(np.square(state2[:,1]))))
		
		#trofal = ((abs(state2[5,0]-state[0])<1.) and (abs(state2[5,1]-state[1])<1.) and (abs(state2[5,2]-state[2]))<1.)
		r = np.sqrt(((state2[self.n,0] - self.x_0)**2) + ((state2[self.n,1]- self.y_0)**2) + ((state2[self.n,2] - self.z_0))**2)/65.
		#r1 = np.sqrt(((state2[5,0] -self.o_0 )**2) + ((state2[5,1]- self.o_1)**2) + ((state2[5,2] - self.o_2))**2)/65.
		#r2 = np.sqrt(((self.x_0 -self.o_0 )**2) + ((self.y_0- self.o_1)**2) + ((self.z_0 - self.o_2))**2)/65.
		
		#reward = -abs(r1-r2)
		#reward = -r
		
		#reward = (np.corrcoef(state3, x_0)[0,1])
		#if r>0.001:
		velo_mag = (np.sqrt(xdot**2 + ydot**2 + zdot**2))**(r)
		reward = exp(-r/(velo_mag))
	
		#else:
			#reward = (r+0.0001)**(-1)#np.exp(-r) #-np.exp(-np.exp(-r))
				
		
		
		
		#reward = -(((state2[2,0]-state[0])**2 ) +((state2[2,1]-state[1])**2) +((state2[2,2]-state[2]))**2) - 1.0)
		
		'''
		if trofal:
			reward = 1
		else:
			reward = -1
		
		'''
		self.state = state2[self.n,0], state2[self.n,1], state2[self.n,2], xdot, ydot, zdot 
		
		#row = [self.t, self.state[0], self.state[1], self.state[2], self.space_a[rhop], self.space_b[sigmap], self.space_c[betap]] # for discrete action space
		row = [self.t, self.state[0], self.state[1], self.state[2], rhop, sigmap, betap, reward, x_0[-1]] #for continous action space
		#self.t +=self.tau	
		self.t = t[-1]
		
		if self.t <= 10.*(self.tau):
			f = open('lorenz_2-3-4_revised.csv', 'r+')
			f.truncate()
			
		with open('lorenz_2-3-4_revised.csv', 'a') as output:
			writer = csv.writer(output)
			writer.writerow(row)
			
		output.close()
			
		return np.array(self.state), reward, done, {}
		
    	  
		
	def reset(self):
		self.state = self.np_random.uniform(low=-40, high=50, size=(6,))
		self.fig.clf() 
		#print(self.n)
		
		return np.array(self.state) 
		
	def render(self, mode='human'):
		
		self.n +=0
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
			

    			
		    
		    
        
        	      
       
					
		
		
