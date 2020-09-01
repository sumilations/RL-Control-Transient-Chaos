import numpy as np
import csv
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

rho   =  28.0
sigma = 10.0
beta  = 8.0/3.0

t, x, y, z, rhop, sigmap, betap = np.loadtxt('lorenz.csv', delimiter=',', unpack=True)

sigmap = sigmap/sigma
rhop   = rhop/rho
betap  = betap/beta

print(np.max(sigmap), np.max(rhop), np.max(betap))
interval = 500
frames = int(len(x)/(interval))
print(frames)

Writer = animation.writers['ffmpeg']
writer = Writer(fps=5, metadata=dict(artist='Me'), bitrate=1800)

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(np.min(sigmap), np.max(sigmap))
ax.set_ylim(np.min(rhop), np.max(rhop))
ax.set_zlim(np.min(betap), np.max(betap))  
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

def animate(i):
	ax.clear()
	ax.set_xlim(np.min(sigmap), np.max(sigmap))
	ax.set_ylim(np.min(rhop), np.max(rhop))
	ax.set_zlim(np.min(betap), np.max(betap))
	ax.plot3D(sigmap[int(100*i):int((i+1)*interval)], rhop[int(100*i):int((i+1)*interval)], betap[int(100*i):int((i+1)*interval)])
	
    
ani = matplotlib.animation.FuncAnimation(fig, animate, frames=frames, repeat=False)
ani.save('lorenz_perturb2.mp4', writer=writer)	

	


