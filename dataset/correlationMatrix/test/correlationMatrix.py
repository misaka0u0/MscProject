import matplotlib.pyplot as plt
from enum import IntEnum
from operator import length_hint
from matplotlib import image
from PIL import Image
from typing import Optional
import numpy as np

from scipy.signal import correlate

num_points = 1
W, H = 400, 400 # x, y axis
Length = 150 # z axis
radius = 2.0 # radius on focal plane
# Vz = 15 # velocity max caculated with V(y, z), flows through x-axis
Zr = 80 # Rayleigh length

intensity = 125 # The max intensity, 'a' in Gaussian blob

win_size = 400
img1 = np.zeros([W, H])

# Generate two arrays the same shape as the image.
# In pixX, every array element has a value equal to the x coordinate of that pixel
# In pixY, every array element has a value equal to the y coordinate of that pixel
pixX, pixY = np.meshgrid(np.arange(W), np.arange(H))

class Point:
    def __init__(self, size: float, dz: Optional[int] = None):
        self.x = 200
        self.y = 200
        self.z = 0
        self.dz = dz if dz is not None else 0
        self.size = size * np.sqrt(1 + ((self.z - self.dz) / Zr) ** 2)  # + 0.02 * self.z
        # self.size = size
        self.intensity = intensity 


def drawPoint(img, point: Point):
    half_size = point.size
    w, h = img.shape
    x = point.x 
    y = point.y 


    #draw particles in blob
    img += point.intensity * ((np.sqrt(2 * np.pi) / half_size) ** 2) * np.exp(-(((x - pixX.T) / half_size)**2 + ((y - pixY.T) / half_size)**2))
    img = np.clip(img, 0, 255)


import os

# Check if the directories exist, and if not, create them
os.makedirs('./object', exist_ok=True)

correlation_stack = []
for dz in range(-Length, Length + 1, 10):  # Iterate over possible dz values
    point_groups = []
    points = [Point(radius, dz=dz) for _ in range(num_points)]  # Initialize points with the current dz
    point_groups.append(points)

    # Initialize the images
    img1 = np.zeros([W, H])

    # Update and draw points
    for points in point_groups:
        for p in points:
            drawPoint(img1, p)

    a_win = img1 [:win_size, :win_size].copy()
    b_win = img1 [:win_size, :win_size].copy()
    cross_corr = correlate(b_win - b_win.mean(), a_win - a_win.mean(), method="fft")

    correlation_stack.append(cross_corr)

    # Save the images
    # Normalize to [0, 1]
    normalized_array = (cross_corr - cross_corr.min()) / (cross_corr.max() - cross_corr.min())

    # Save as a 32-bit floating point TIFF image
    Image.fromarray(normalized_array.T.astype(np.float32)).save(f'./object/dz_{dz + 150}.tif')
   

# Convert the list of 2D arrays into a 3D numpy array
correlation_stack = np.stack(correlation_stack)


np.save('corrMatrix.npy', correlation_stack)
# data = np.load('filename.npy')

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
Y, X = np.meshgrid(np.arange(correlation_stack.shape[1]), np.arange(correlation_stack.shape[2]))

ax.plot_surface(Y, X, correlation_stack[0].T, cmap='jet', linewidth=0.2)  # type: ignore
plt.title("Correlation map — peak is the most probable shift")
plt.show()
