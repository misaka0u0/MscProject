# import matplotlib.pyplot as plt
from enum import IntEnum
from operator import length_hint
import numpy as np
from PIL import Image
from typing import Optional



num_points = 1
W, H = 20, 20 # x, y axis
Length = 150 # z axis
radius = 2.0 # radius on focal plane
# Vz = 15 # velocity max caculated with V(y, z), flows through x-axis
Zr = 80 # Rayleigh length

intensity = 125 # The max intensity, 'a' in Gaussian blob

img1 = np.zeros([W, H])

# Generate two arrays the same shape as the image.
# In pixX, every array element has a value equal to the x coordinate of that pixel
# In pixY, every array element has a value equal to the y coordinate of that pixel
pixX, pixY = np.meshgrid(np.arange(W), np.arange(H))

class Point:
    def __init__(self, size: float, dz: Optional[int] = None):
        self.x = 10
        self.y = 10
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
os.makedirs('./PSF', exist_ok=True)

image_stack = []
for dz in range(-Length*2, Length*2 + 1, 10):  # Iterate over possible dz values
    np.random.seed(42)
    point_groups = []
    points = [Point(radius, dz=dz) for _ in range(num_points)]  # Initialize points with the current dz
    point_groups.append(points)

    # Initialize the images
    img1 = np.zeros([W, H])


    # Update and draw points
    for points in point_groups:
        for p in points:
            drawPoint(img1, p)

    image_stack.append(img1)
    Image.fromarray(img1.T.astype(np.uint8)).save(f'./PSF/dz_{dz+300}.bmp')

# Convert the list of 2D arrays into a 3D numpy array
image_stack_3d = np.stack(image_stack)

# print(image_stack1)
print(np.shape(image_stack))
print(np.shape(image_stack_3d))

np.save('PSF.npy', image_stack_3d)
# data = np.load('filename.npy')



