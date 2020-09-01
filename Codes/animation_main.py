import numpy as np
import csv
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

rho   =  28.0
sigma = 10.0
beta  = 8.0/3.0

t, x, y, z, sigmap, rhop, betap = np.loadtxt('lorenz.csv', delimiter=',', unpack=True)
t = t[150000:]
x = x[150000:]
y = y[150000:]
z = z[150000:]

interval = 600
frames = int(len(x)/(interval))
print(frames, len(x))

Writer = animation.writers['ffmpeg']
writer = Writer(fps=5, metadata=dict(artist='Me'), bitrate=1800)

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(np.min(x), np.max(x))
ax.set_ylim(np.min(y), np.max(y))
ax.set_zlim(np.min(z), np.max(z))  
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

def animate(i):
	#ax.clear()
	ax.set_title(label='Control ON', loc='left')
	ax.set_xlim(np.min(x), np.max(x))
	ax.set_ylim(np.min(y), np.max(y))
	ax.set_zlim(np.min(z), np.max(z))
	ax.plot3D(x[int(100*i):int((i+1)*interval)], y[int(100*i):int((i+1)*interval)], z[int(100*i):int((i+1)*interval)])
	
    
ani = matplotlib.animation.FuncAnimation(fig, animate, frames=frames, repeat=False)
ani.save('lorenz4.mp4', writer=writer)	

	


