import matplotlib
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1.inset_locator import (mark_inset, zoomed_inset_axes)
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.patches as patches
from curlyBrace import curlyBrace




datafile = np.loadtxt("/home/user/Desktop/lorenz_transient_terminal_full6.csv", delimiter = ",") #np.loadtxt("/home/user/Desktop/lorenz_transient_terminal__test_full6.csv", delimiter = ",")
##
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



font = {'family': 'Times New Roman',
        'color':  'k',
        'weight': None,
        'style': 'normal',
        'size': 10,
        }


k_r2 = 0.2

str_text = 'Transiently-Chaotic'
str_text2 = 'Chaotic'
rho = 20
sigma = 10
beta  = 8/3.0
x0 = 1230
y0 = 40
width = 700
height = 0
facecol = 'cyan'
edgecol = 'black'
linewidth=1

x1 = 2000
y1 = 40
width1 = 690
p1 = [x0, y0]
p2 = [x0+width, y0]

p3 = [x1, y1]
p4 = [x1+width1, y0]

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
fig, ax = plt.subplots(3, 1, sharex=True, sharey=False, gridspec_kw={'hspace': 0, 'wspace': 0})
ax[0].scatter(t[::3],pert_rho[::3]+rho,s=2,color="blue")
ax[1].scatter(t[::3],pert_sig[::3]+sigma,s=2,color = "forestgreen")
ax[2].scatter(t[::3],pert_beta[::3]+beta,s=2, color="black")


#ax[0].set_yticks([])
ax[2].set_xlabel("Training time",fontname="Times New Roman",size=15)
ax[0].set_ylabel("$\\rho$",size=15)
ax[2].set_ylabel("$\\beta$",size=15)
ax[1].set_ylabel("$\\sigma$",size=15)


plt.xlim(1200,2700)
plt.tight_layout()			    
plt.savefig('pert_transition.pdf', bbox_inches='tight' )
plt.show()
