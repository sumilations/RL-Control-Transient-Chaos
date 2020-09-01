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
fig, ax = plt.subplots(figsize=[4, 3])

#scatter = ax.scatter(x, z, c = pert_rho, s = 2, cmap = cm.seismic)
#ax.text2D(0.075, 0, "$\mathrm{Z}$", ha='center', color ='black', size=10)
#ax.text2D(0.051, -0.067, "$\mathrm{Y}$", ha='center', color ='Black', size=10)
#ax.text2D(-0.04, -0.08, "$\\bf{\mathrm{X}}$", ha='center', color ='Black', size=10)
#ax.text(1550, 44, "Non-Chaotic", ha='center', color ='red', fontweight='bold', size=10)
#ax.text(2300, 44, "Chaotic", ha='center', color ='red', fontweight='bold', size=10)

#ax.set_title("Training of the RL network")
ax.grid(False)
#arr1 = patches.FancyBboxPatch((x0,y0),width,height,boxstyle='darrow',
                              #lw=linewidth,ec=edgecol,fc=facecol)
                             
#arr2 = patches.FancyBboxPatch((x1,y1),width1,height,boxstyle='darrow', lw=linewidth,ec=edgecol,fc=facecol)
#t2 = matplotlib.transforms.Affine2D().rotate_deg_around(x1,y1,0) + ax.transData
#arr2.set_transform(t2) # Rotate the arrow
'''
plt.arrow(x0, y0, width, 0, head_width=2, head_length=50, linewidth=1, color='k', length_includes_head=True)  
plt.arrow(x0+width, y0, -width, 0, head_width=2, head_length=50, linewidth=1, color='k', length_includes_head=True)      

plt.arrow(x1, y1, width, 0, head_width=2, head_length=50, linewidth=1, color='k', length_includes_head=True)  
plt.arrow(x1+width, y1, -width, 0, head_width=2, head_length=50, linewidth=1, color='k', length_includes_head=True) 
'''
'''
fs=10
ax.annotate('SDL', xy=(0.5, 0.90), xytext=(0.5, 1.00), xycoords='axes fraction', 
            fontsize=fs*1.5, ha='center', va='bottom',
            bbox=dict(boxstyle='square', fc='white'),
            arrowprops=dict(arrowstyle='-[, widthB=7.0, lengthB=1.5', lw=2.0))
'''
                
ax.plot(t, x, linewidth = 0.3, linestyle ="-", color = "blue")
curlyBrace(fig, ax, p1, p2, k_r2, bool_auto=True, str_text=str_text, color='r', lw=2, ls='-.', int_line_num=1, fontdict=font)
curlyBrace(fig, ax, p3, p4, k_r2, bool_auto=True, str_text=str_text2, color='r', lw=2, ls='-.', int_line_num=1, fontdict=font)

#ax.set_title('$\epsilon/D = {}, D={}, f_d = {}$'.format(ratio, D, f1))
#ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel("Training time",fontname="Times New Roman")
ax.set_ylabel("x",fontname="Times New Roman")
#ax.add_patch(arr1)
#ax.add_patch(arr2)

#ax.set_zticks([])
plt.xlim(1200,2700)
plt.ylim(-50,60)


#plt.colorbar(scatter)
#ax.plot(t, vel_mag, linewidth=2)
#print(np.min(vel_mag), np.max(vel_mag))
#surf = ax.plot_surface(x, y, z, rstride=1, cstride=1)
#ax.set_title("(a)")
#ax.set_xlim(-20,20)
#ax.set_ylim(-20,20)
#ax.set_zlim(0,40)

#ax.set_xlabel("training time")
#ax.set_ylabel("$\mathrm{V_{mag}}$")
#plt.tight_layout()
#ax.set_zlabel("Z")

'''
ax = fig.add_subplot(122)#, projection='3d')
ax.plot(t, x)
#ax.quiver(x, y, z, vel_x, vel_y, vel_z, length=3,  normalize=True)

#ax.set_xlim(-20,20)
#ax.set_ylim(-20,20)
#ax.set_zlim(0,40)
'''
'''
ax.set_xlabel("training time")
ax.set_ylabel("X")

#ax.set_zlabel("Z")
'''
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
plt.tight_layout()			    
plt.savefig('Training_x.pdf', bbox_inches='tight' )
plt.show()
