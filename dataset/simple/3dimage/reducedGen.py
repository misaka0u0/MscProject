# import matplotlib.pyplot as plt
from enum import IntEnum
from operator import length_hint
import numpy as np
from PIL import Image
from typing import Optional
import tifffile



num_points = 1200
W, H = 320, 320 # x, y axis #  500* 300  / (64 * 72 ) = 
DEPTH = 150 # z axis
radius = 2.0 # radius on focal plane
Vmax = 8
# Vz = 15 # velocity max caculated with V(y, z), flows through x-axis
Zr = 80 # Rayleigh length

intensity = 125 # The max intensity, 'a' in Gaussian blob

img1 = np.zeros([H, W])
img2 = np.zeros([H, W])

# Generate two arrays the same shape as the image.
# In pixX, every array element has a value equal to the x coordinate of that pixel
# In pixY, every array element has a value equal to the y coordinate of that pixel
pixY, pixX = np.meshgrid(np.arange(H), np.arange(W))

class Point:
    def __init__(self, size: float, vx: Optional[int] = None, vy: Optional[int] = None, dz: Optional[int] = None):
        self.x = np.random.randint(int(size), int(W - size))
        self.y = np.random.randint(int(size), int(H - size))
        self.z = np.random.randint(-int(DEPTH), int(DEPTH))
        # (1 - (abs(self.y - 160) / 160 ) ** 2 - (abs(self.z) / 150) ** 2)
        self.vx = vx if vx is not None else Vmax * (1 - (abs(self.z) / 150) ** 2)
        self.vy = vy if vy is not None else 0
        self.dz = dz if dz is not None else 0
        self.size = size * np.sqrt(1 + ((self.z - self.dz) / Zr) ** 2)  # + 0.02 * self.z
        # self.size = size
        self.intensity = intensity 
        


    def update(self):
        self.x = (self.x + self.vx) % W
        self.y = (self.y + self.vy) % H



def drawPoint(img, point: Point):
    half_size = point.size
    size = 2 * half_size + 1
    w, h = img.shape
    x = point.x 
    y = point.y 


    #draw particles in blob
    img += point.intensity * ((np.sqrt(2 * np.pi) / half_size) ** 2) * np.exp(-(((x - pixX.T) / half_size)**2 + ((y - pixY.T) / half_size)**2))
    img = np.clip(img, 0, 255)

import os

# Check if the directories exist, and if not, create them
os.makedirs('./image_stack1', exist_ok=True)
os.makedirs('./image_stack2', exist_ok=True)



image_stack1 = []
image_stack2 = []
for dz in range(-DEPTH, DEPTH + 1, 10):  # Iterate over possible dz values
    np.random.seed(42)
    point_groups = []
    points = [Point(radius, dz=dz) for _ in range(num_points)]  # Initialize points with the current dz
    point_groups.append(points)

    # Initialize the images
    img1 = np.zeros([H, W])
    img2 = np.zeros([H, W])


    # Update and draw points
    for points in point_groups:
        for p in points:
            drawPoint(img1, p)
            p.update()
            drawPoint(img2, p)
    image_stack1.append(img1)
    image_stack2.append(img2)


    # Save the images
    Image.fromarray(img1.astype(np.uint8)).save(f'./image_stack1/dz_{dz+150}.bmp')
    Image.fromarray(img2.astype(np.uint8)).save(f'./image_stack2/dz_{dz+150}.bmp')

# Convert the list of 2D arrays into a 3D numpy array
image_stack1 = np.stack(image_stack1)
image_stack2 = np.stack(image_stack2)


np.save('stkA.npy', image_stack1)
np.save('stkB.npy', image_stack2)
# data = np.load('filename.npy')

# np.savetxt('stk1.txt', image_stack1)
# np.savetxt('stk3d.txt', image_stack_3d)
# np.savetxt only works for 1d or 2d array,
# not a stack of 2d arrays or 3d



sideraw1 = np.max(image_stack1[:,:,10:40], axis=2)
sideraw2 = np.max(image_stack2[:,:,10:40], axis=2)
tifffile.imwrite('sideraw.tif', np.array([sideraw1, sideraw2]))