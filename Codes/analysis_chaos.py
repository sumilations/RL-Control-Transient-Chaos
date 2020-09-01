import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

datafile = np.loadtxt("/home/user/Desktop/lorenz_transient_terminal__test_full2.csv", delimiter = ',')
x = datafile[:,1]
y = datafile[:,8]
z = datafile[:,7]  # first derivative

cmap = ListedColormap(['r', 'g', 'b'])
norm = BoundaryNorm([-1, -0.5, 0.5, 1], cmap.N)

# Create a set of line segments so that we can color them individually
# This creates the points as a N x 1 x 2 array so that we can stack points
# together easily to get the segments. The segments array for line collection
# needs to be numlines x points per line x 2 (x and y)
points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# Create the line collection object, setting the colormapping parameters.
# Have to set the actual values used for colormapping separately.
lc = LineCollection(segments, cmap=cmap, norm=norm)
lc.set_array(z)
lc.set_linewidth(3)
plt.gca().add_collection(lc)

plt.xlim(x.min(), x.max())
plt.ylim(-1.1, 1.1)
plt.show()


###Is that stochastic-resonance?
