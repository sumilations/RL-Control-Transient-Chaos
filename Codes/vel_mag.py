import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.cm as cm
from matplotlib.patches import Circle, Ellipse
from mpl_toolkits.axes_grid1.inset_locator import (mark_inset, zoomed_inset_axes)
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d

import numpy as np

datafile = np.loadtxt("/home/user/Desktop/lorenz_transient_terminal__test_full6.csv", delimiter = ",")
#np.loadtxt("/home/user/Desktop/lorenz_transient_terminal_full6.csv", delimiter = ",")#
nmax = len(datafile[:,1])

t            =         datafile[0:nmax,0]
x            =         datafile[0:nmax,1]
y            =         datafile[0:nmax,2]
z            =         datafile[0:nmax,3]
#r1           =         datafile[0:nmax,13]
vel_x        =         datafile[0:nmax,8]
vel_y        =         datafile[0:nmax,9]
vel_z        =         datafile[0:nmax,10]
vel_mag      =         datafile[0:nmax,10]
pert_rho     =         datafile[0:nmax,4]
pert_sig     =         datafile[0:nmax,5]
pert_beta    =         datafile[0:nmax,6]
pert_rms     =         datafile[0:nmax,7]
vel_x = vel_x/(np.sqrt(vel_x**2 + vel_z**2))
vel_z = vel_z/(np.sqrt(vel_x**2 + vel_z**2))

lm= []
velm=[]
lp = []
velp = []


'''
for i in range(0,nmax):
	if pert_rho[i] != -2.0:
		pert_rho[i] = 0
	#if pert_rho[i] + pert_sig[i] + pert_beta[i] < 0:
		#pert_rms[i] = -pert_rms[i]
'''	
'''
var_list     =      [x, y, z, r1]
vel_list     =      [vel_x, vel_y, vel_z, vel_mag]
pert_list    =      [pert_rho, pert_sig, pert_beta, pert_rms]
v_label      =      ["$v_x$","$v_y$", "$v_z$", "$v_r$"]
x_label      =      ["x","y","z", "r"]
pert_label   =      ["$\Delta\\rho$","$\Delta\sigma$","$\Delta\\beta$", "$pert_{rms}$"]
'''


for i in range(0,len(t)):
	if pert_rho[i]<0:
		lm.append([x[i],z[i]])
		velm.append([vel_x[i], vel_z[i]])
		
for i in range(0,len(t)):
	if pert_rho[i]>0:
		lp.append([x[i],z[i]])
		velp.append([vel_x[i], vel_z[i]])
		
lm=np.array(lm)		
velm=np.array(velm)		
lp=np.array(lp)		
velp=np.array(velp)		
		
#fig, ax = plt.subplots(1, 1, sharex='col', sharey='row',  gridspec_kw={'hspace': 0, 'wspace': 0})
fig = plt.figure(figsize=(5,3))
ax = fig.add_subplot(111, projection = '3d')
#p = Ellipse((10, 35), width=20, height=19, angle=-45, edgecolor='k', fc='none', ls='--')
#ax.add_patch(p)
#art3d.pathpatch_2d_to_3d(p, z=-20, zdir="y")
#fig, ax = plt.subplots(figsize=[5, 4])
s=50
#ax.quiver(x[::s],z[::s],vel_x[::s],vel_z[::s], angles='xy', scale_units='xy', scale=0.5, pivot='mid', color='k', linewidths=0.5, edgecolors='k', headwidth=4)
scatter = ax.scatter(x,y,z, c = pert_rho, s = 0.7, cmap = cm.seismic)
ax.text2D(0.075, 0, "$\mathrm{z}$", ha='center', color ='black', size=10)
ax.text2D(0.051, -0.067, "$\mathrm{y}$", ha='center', color ='Black', size=10)
ax.text2D(-0.04, -0.08, "$\\bf{\mathrm{x}}$", ha='center', color ='Black', size=10)
ax.text(0, 0, 37, "$\Re$", ha='center', color ='k', fontweight='bold', size=10)
#ax.set_title("$\\Delta\\rho < 0$ when $V_{z}<0$")
#ax.grid(False)
#ax.plot(t0, vel)
#ax.set_title('$\epsilon/D = {}, D={}, f_d = {}$'.format(ratio, D, f1))
ax.set_xticks([])
ax.set_yticks([])
#ax.set_xlabel("X")
#ax.set_ylabel("Z")

ax.set_zticks([])
#plt.xlim(-20,20)
#plt.ylim(0,40)


#plt.colorbar(scatter)
#ax.plot(t, vel_mag, linewidth=2)
#print(np.min(vel_mag), np.max(vel_mag))
#surf = ax.plot_surface(x, y, z, rstride=1, cstride=1)
#ax.set_title("(a)")
#ax.set_xlim(-20,20)
#ax.set_ylim(-20,20)
#ax.set_zlim(0,40)
'''
axins = zoomed_inset_axes(ax, 2.5, loc=9)

x1, x2, y1, y2 = -20, 4, 15, 25
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.set_xticklabels('')
axins.set_yticklabels('')
axins.set_xticks([])
axins.set_yticks([])

box,c1,c2 = mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="0.5")
plt.setp(box, linewidth=1, color="black")
plt.setp([c1,c2], linestyle=":", color ="black")

axins.quiver(x[::10],z[::10],vel_x[::10],vel_z[::10], angles='xy', scale_units='xy', scale=2, pivot='mid', color='k', linewidths=0.5, edgecolors='k', headwidth=8)
'''
#ax.set_xlabel("training time")
#ax.set_ylabel("$\mathrm{V_{mag}}$")
#plt.tight_layout()
#ax.set_zlabel("Z")


#ax = fig.add_subplot(122)#, projection='3d')
#ax.plot(t, x)
#ax.quiver(x, y, z, vel_x, vel_y, vel_z, length=3,  normalize=True)

ax.set_xlim(-20,20)
ax.set_ylim(-20,20)
ax.set_zlim(0,40)
'''
'''
#ax.set_xlabel("training time")
#ax.set_ylabel("X")
#plt.tight_layout()
#ax.set_zlabel("Z")

'''
for i in range(3, 4):
	for j in range(3, 4):
		for k in range(3, 4):
			ax.scatter(var_list[i],  vel_list[j], s=1, c=pert_list[k], cmap = cm.seismic, label = pert_label[k])
			ax.legend()
			
			if k==0:
				ax[j,k].set_ylabel(v_label[j])
				
			if j==2:
				ax[j,k].set_xlabel(x_label[0])
				
'''				    
plt.savefig('pert_rho_coord.pdf', bbox_inches='tight')
plt.show()
