import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the function f(x, y)
def f(x, y):
    return (1 - (abs(y - 185) / 185 ) ** 2 - (abs(x) / 150) ** 2)

# Generate x and y values
x = np.linspace(-150, 150, 3000)
y = np.linspace(0, 369, 3690)
X, Y = np.meshgrid(x, y)

# Evaluate the function f(x, y) for each pair of (x, y) values
Z = f(X, Y)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surface = ax.plot_surface(X, Y, Z, cmap='viridis')

fig.colorbar(surface)
# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('f(X, Y)')
ax.set_title('3D Plot')

# Show the plot
plt.show()


