import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1.inset_locator import (mark_inset, zoomed_inset_axes)
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

datafile = np.loadtxt("/home/user/Desktop/lorenz_transient_terminal__test_full6.csv", delimiter = ",")
#np.loadtxt("/home/user/Desktop/lorenz_transient_terminal_full6.csv", delimiter = ",")#
nmax = len(datafile[:,1])

t            =         datafile[0:nmax,0]
x            =         datafile[0:nmax,1]
y            =         datafile[0:nmax,2]
z            =         datafile[0:nmax,3]
#r1           =        datafile[0:nmax,13]
vel_x        =         datafile[0:nmax,8]
vel_y        =         datafile[0:nmax,9]
vel_z        =         datafile[0:nmax,10]
vel_mag      =         datafile[0:nmax,10]
pert_rho     =         datafile[0:nmax,4]
pert_sig     =         datafile[0:nmax,5]
pert_beta    =         datafile[0:nmax,6]
pert_rms     =         datafile[0:nmax,7]
norm = 1#(np.sqrt(vel_x**2 + vel_y**2 + vel_z**2))
vel_x = vel_x/norm
vel_y = vel_y/norm
vel_z = vel_z/norm

lm= []
velm=[]
lp = []
velp = []

for i in range(0,len(t)):
	if pert_rho[i]<0:
		lm.append([x[i],y[i],z[i]])
		velm.append([vel_x[i], vel_y[i], vel_z[i]])
		
for i in range(0,len(t)):
	if pert_rho[i]>0:
		lp.append([x[i],y[i],z[i]])
		velp.append([vel_x[i], vel_y[i], vel_z[i]])
				
				
lm=np.array(lm)		
velm=np.array(velm)		
lp=np.array(lp)		
velp=np.array(velp)		
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection = '3d')
scatter = ax.scatter(x, y, z, c = pert_beta, s = 4, cmap = cm.seismic)

ax.grid()
print(np.shape(lm[:,0]))
'''
s = 25
ax.quiver(lm[:,0][::s], lm[:,1][::s], lm[:,2][::s], velm[:,0][::s], velm[:,1][::s], velm[:,2][::s], normalize =True, length = 2, color = "r")
ax.quiver(lp[:,0][::s], lp[:,1][::s], lp[:,2][::s], velp[:,0][::s], velp[:,1][::s], velp[:,2][::s], normalize =True, length = 2, color = "k")
'''
#ax.axis("equal")
ax.set_xlim(-20,20)
ax.set_ylim(-20,20)
ax.set_zlim(0,40)

ax.set_aspect("equal")

ax.text2D(0.075, 0, "$\mathrm{Z}$", ha='center', color ='black', size=18)
ax.text2D(0.051, -0.067, "$\mathrm{Y}$", ha='center', color ='Black', size=18)
ax.text2D(-0.04, -0.08, "$\\bf{\mathrm{X}}$", ha='center', color ='Black', size=18)
#ax.text(-7.12, -7.12, 21, "$\\bf{\mathrm{P_{-}}}$", ha='center', color ='red', fontweight='bold', size=10)

#ax.grid(True)
#ax.plot(t0, vel)
#ax.set_title('$\epsilon/D = {}, D={}, f_d = {}$'.format(ratio, D, f1))
#ax.set_xticks([])
#ax.set_yticks([])
#ax.set_zticks([])

ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_zticklabels([])


plt.savefig("attractor3d_pert_rho.png")
plt.show()
