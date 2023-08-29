import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the function f(z, y)
def f(z, y):
    return 20 * (1 - (abs(y - 185) / 185 ) ** 2 - (abs(z) / 150) ** 2)

# Generate x and y values
z = np.linspace(-150, 150, 3000)
y = np.linspace(0, 369, 3690)
Z, Y = np.meshgrid(z, y)

# Evaluate the function f(x, y) for each pair of (x, y) values
V = f(Z, Y)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surface = ax.plot_surface(Z, Y, V, cmap='viridis')

fig.colorbar(surface)
# Set labels and title
ax.set_xlabel('Z')
ax.set_ylabel('Y')
ax.set_zlabel('f(Z, Y)')
ax.set_title('3D Plot')

# Show the plot
plt.show()


