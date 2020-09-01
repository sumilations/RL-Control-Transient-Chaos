import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import cnames
from matplotlib import animation
import cool_colors
import sys


rho   =  20.0
sigma = 10.0
beta  = 8.0/3.0

N_trajectories = 1
datafile = np.loadtxt("lorenz_c.csv")

nmax = len(datafile[:,1])
print(nmax)

t            =         datafile[0:nmax,0]
x            =         datafile[0:nmax,1]
y            =         datafile[0:nmax,2]
z            =         datafile[0:nmax,3]
rhop         =         datafile[0:nmax,7]

rhop = rhop/rho
#start_index = 50000
#finish_index = 150000

#print(finish_index)
#t = t[start_index:finish_index]
#x = x[start_index:finish_index]
#y = y[start_index:finish_index]
#z = z[start_index:finish_index]

#sigmap = sigmap[start_index:finish_index]/sigma
#rhop = rhop[start_index:finish_index]/rho
#betap = betap[start_index:finish_index]/beta

sigmap=np.zeros(len(rhop))
betap= np.zeros(len(rhop))

x_ = np.column_stack((x,y,z))
x_t = np.asarray([x_])
interval = 100
frames = int(len(x)/(interval))

# Set up figure & 3D axis for animation
fig = plt.figure()
fig.set_facecolor('black')
ax = fig.add_axes([0, 0, 1, 1], projection='3d')

# choose a different color for each trajectory
colors1 = plt.cm.cool(np.linspace(0.9, 1, N_trajectories))
colors2 = plt.cm.jet(np.linspace(0.9, 1, N_trajectories))
cmap=plt.get_cmap('cool')

def animate(i):
    # we'll step two time-steps per frame.  This leads to nice results.
    #i = (2 * i) % x_t.shape[1]
    
    ax.axis('off')
    ax.set_facecolor('black')
    sigma_text = ax.text2D(0.225, 0.95, " ", color='white', transform=ax.transAxes)
    rho_text = ax.text2D(0.425, 0.95, " ", color='white', transform=ax.transAxes)
    ax.text(7.12, 7.12, 20, "$\mathrm{{P}_{+}}$", ha='center', color ='red', size=12)
    ax.text(-7.12, -7.12, 20, "$\mathrm{{P}_{-}}$", ha='center', color ='red', size=12)
    beta_text = ax.text2D(0.625, 0.95, " ", color ='white', transform=ax.transAxes)
    control_text = ax.text2D(0.055, 0.95, "Control: ", color ='white', size = 12, transform=ax.transAxes)
    control_text_on = ax.text2D(0.16, 0.95, " ", color ='Green', size=12, transform=ax.transAxes)
    control_text_off = ax.text2D(0.16, 0.95, " ", color ='Red', size=12, transform=ax.transAxes)
    #Target_text = ax.text(0, 0, -5, "Target", ha='center', color ='Red', size=12)

    # prepare the axes limits
    ax.set_xlim((-20, 20))
    ax.set_ylim((-30, 30))
    ax.set_zlim((5, 55))

    lines = sum([ax.plot([], [], [], '-', c=c) for c in colors1], [])
    pts = sum([ax.plot([], [], [], 'o', c=c, linewidth=1000) for c in colors2], [])
    
    for line, pt in zip(lines, pts):
        pt.set_data([], [])
        pt.set_3d_properties([])
    
    # set point-of-view: specified by (altitude degrees, azimuth degrees)
    ax.view_init(30, 0)
	
    currentDataIndex = (i)*interval
    alpha_vec = np.linspace(0.1,1.0,currentDataIndex)
    linewidth_vec = np.linspace(0.01,0.4,currentDataIndex)
    linewidth_vec = np.exp(linewidth_vec)
    points = np.array([x[:currentDataIndex], y[:currentDataIndex], z[:currentDataIndex]]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    colors=[cmap(float(ii)/(currentDataIndex-1)) for ii in range(currentDataIndex-1)]

    for ii in range( currentDataIndex-1):
        segii=segments[ii]
        lii,=ax.plot(segii[:,0],segii[:,1],segii[:,2],color=colors[-ii-1],linewidth=linewidth_vec[ii], alpha=alpha_vec[ii])
        xa=segii[:,0]
        ya=segii[:,1]
        za=segii[:,2]


    pt.set_data(xa[-1:], ya[-1:])
    pt.set_3d_properties(za[-1:])
    
    #sigma_text.set_text('$\Delta\sigma$ = %.3f' %sigmap[currentDataIndex])
    #beta_text.set_text('$\Delta \\beta $ = %.3f' %betap[currentDataIndex])
    rho_text.set_text('$\Delta\\rho$ = %.3f' %rhop[currentDataIndex])
    
    
    if i< int(2.0*frames/3.0) :
        control_text_on.set_text('ON')
        control_text_off.set_text('')
    else:
        control_text_on.set_text('')
        control_text_off.set_text('OFF')	

    ax.view_init(30, 0.6 * i)
    fig.canvas.draw()
    plt.savefig('T_{0:06}.png'.format(i))
    #print(points, alpha_vec)

    ax.cla()

    return points, segments, linewidth_vec, alpha_vec# lines + pts, sigma_text

# instantiate the animator.
anim = animation.FuncAnimation(fig, animate, frames=frames, interval=interval, blit=False)

# Save as mp4. This requires mplayer or ffmpeg to be installed
anim.save('lorentz_attractor.mp4', fps=15, extra_args=['-vcodec', 'libx264'])
#mywriter = animation.FFMpegWriter(fps=30)
#anim.save('ani.mp4', writer=mywriter, dpi=600)

n1 = 1
n2 =100
"""
for i in range(n1,n2):
    i
    animate(i)
"""
