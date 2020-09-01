from matplotlib.image import NonUniformImage
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm


data =  np.loadtxt("/home/user/Desktop/lorenz_transient_terminal__test_full5.csv", delimiter = ",")

rho   =  20.00
sigma = 10.00
beta  = 8.0/3.0

dimension = 0

drho = data[dimension:,4]
dsigma = data[dimension:,5]
dbeta = data[dimension:,6]

drho_edges = np.linspace(-rho/10, rho/10, 10)
dsigma_edges = np.linspace(-sigma/10, sigma/10, 10)
dbeta_edges = np.linspace(-beta/10, beta/10, 10)

H, xedges, yedges = np.histogram2d(dsigma, drho, bins=(dsigma_edges, drho_edges))
H = H.T 

fig = plt.figure(figsize=(7, 3))

ax = fig.add_subplot(111, title='P(sigma-rho)', aspect='equal', xlim=xedges[[0, -1]], ylim=yedges[[0, -1]])
im = NonUniformImage(ax, interpolation='bilinear', cmap = cm.Greys_r)
xcenters = (xedges[:-1] + xedges[1:]) / 2
ycenters = (yedges[:-1] + yedges[1:]) / 2
im.set_data(xcenters, ycenters, H)
ax.images.append(im)
plt.show()
