import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.cm as cm
import numpy as np


datafile = np.loadtxt("/home/user/Desktop/lorenz_transient_terminal__test_full5.csv", delimiter = ",")

nmax = len(datafile[:,1])


x            =         datafile[0:nmax,1]
y            =         datafile[0:nmax,2]
z            =         datafile[0:nmax,3]
vel_x        =         datafile[0:nmax,8]
vel_y        =         datafile[0:nmax,9]
vel_z        =         datafile[0:nmax,10]
vel_mag      =         datafile[0:nmax,13]
pert_rho     =         datafile[0:nmax,4]
pert_sig     =         datafile[0:nmax,5]
pert_beta    =         datafile[0:nmax,6]
pert_rms     =         datafile[0:nmax,7]

var_list     =         [x, y, z]
vel_list     =         [vel_x, vel_y, vel_z]
pert_list    =         [pert_rho, pert_sig, pert_beta]

v_label     =      ["$v_x$","$v_y$", "$v_z$"]
x_label      =      ["x","y","z"]
pert_label   =      ["$\Delta\\rho$","$\Delta\sigma$","$\Delta\\beta$"]
fig, ax = plt.subplots(1, 1, sharex='col', sharey='row',  gridspec_kw={'hspace': 0, 'wspace': 0})


for i in range(1, 2):
	for j in range(2, 3):
		for k in range(1, 2):
			ax.scatter(var_list[i],  vel_list[j], s=1, c=pert_list[k], cmap = cm.seismic, label = pert_label[k])
			#ax[j,k].legend()
			"""
			if k==0:
				ax[j,k].set_ylabel(v_label[j])
				
			if j==2:
				ax[j,k].set_xlabel(x_label[0])
			"""	
				    
plt.savefig('mod_all_1.png')
plt.show()


fig1, ax1 = plt.subplots(3, 3, sharex='col', sharey='row', gridspec_kw={'hspace': 0, 'wspace': 0})


for i in range(1, 2):
	for j in range(0, 3):
		for k in range(0, 3):
			ax1[j,k].scatter(var_list[i],  vel_list[j], s=1, c=pert_list[k], cmap = cm.seismic, label = pert_label[k])
			ax1[j,k].legend()
			
			if k==0:
				ax1[j,k].set_ylabel(v_label[j])
				
			if j==2:
				ax1[j,k].set_xlabel(x_label[1])		


plt.savefig('mod_all_2.png')
plt.show()

fig2, ax2 = plt.subplots(3, 3, sharex='col', sharey='row', gridspec_kw={'hspace': 0, 'wspace': 0})


for i in range(2, 3):
	for j in range(0, 3):
		for k in range(0, 3):
			ax2[j,k].scatter(var_list[i],  vel_list[j], s=1, c=pert_list[k], cmap = cm.seismic, label = pert_label[k])
			ax2[j,k].legend()
			
			if k==0:
				ax2[j,k].set_ylabel(v_label[j])
				
			if j==2:
				ax2[j,k].set_xlabel(x_label[2])


plt.savefig('mod_all_3.png')
plt.show()



  
           
'''
f, ax = plt.subplots(1,2, sharex=True, sharey=True)
#ax[0].tripcolor(x,y,z)
contour1 = ax[0].tricontourf(x,y,vel, 60, cmap=cm.Greys_r) # choose 20 contour levels, just to show how good its interpolation is
plt.colorbar(contour1)
contour2 = ax[1].tricontourf(x,y,pert, 60, cmap=cm.Greys_r) # choose 20 contour levels, just to show how good its interpolation is
plt.colorbar(contour2)
'''
#ax[1].plot(x,y, 'ko ')
#ax[0].plot(x,y, 'ko ')
#plt.colorbar(contour2)

