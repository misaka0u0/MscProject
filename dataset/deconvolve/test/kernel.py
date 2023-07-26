import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

Z = np.linspace(-200, 200, 1000)

r = 4*np.sqrt(1+(Z/80)**2)
theta = np.linspace(0, 2.*np.pi, 200)

# Create a 2D grid of x and y values
Z, theta = np.meshgrid(Z, theta)
X = r * np.cos(theta)
Y = r * np.sin(theta)

# X = 4 * np.sqrt(1 + (Z / 80) ** 2)
# Y = 4 * np.sqrt(1 + (Z / 80) ** 2)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_zlim(-150,150)
surf = ax.plot_surface(X, Y, Z, rstride=20, cstride=20, color='k', edgecolors='w')
# ax.view_init(60, 35)


# fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

# Plot the surface
# ax.plot(X, Y, Z)

# Set labels and title
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('3D Plot')

# Show the plot
plt.show()









