# import matplotlib.pyplot as plt
from enum import IntEnum
from operator import length_hint
import numpy as np
from PIL import Image
from typing import Optional

num_groups = 1
num_points = 800
W, H = 511, 369 # x, y axis
Length = 150 # z axis
radius = 2.0 # radius on focal plane
Vmax = 20
# Vz = 15 # velocity max caculated with V(y, z), flows through x-axis
Zr = 80 # Rayleigh length

intensity = 125 # The max intensity, 'a' in Gaussian blob

img1 = np.zeros([W, H])
img2 = np.zeros([W, H])

# Generate two arrays the same shape as the image.
# In pixX, every array element has a value equal to the x coordinate of that pixel
# In pixY, every array element has a value equal to the y coordinate of that pixel
pixX, pixY = np.meshgrid(np.arange(W), np.arange(H))

class Point:
    def __init__(self, size: float, vx: Optional[int] = None, vy: Optional[int] = None):
        self.x = np.random.randint(int(size), int(W - size))
        self.y = np.random.randint(int(size), int(H - size))
        self.z = np.random.randint(-int(Length), int(Length))
        self.vx = vx if vx is not None else Vmax * (1 - (abs(self.y - 185) / 185 ) ** 2 - (abs(self.z) / 150) ** 2)
        self.vy = vy if vy is not None else 0
        self.size = size * np.sqrt(1 + (self.z / Zr) ** 2)  # + 0.02 * self.z
        # self.size = size
        self.intensity = intensity 
        
        # self.vx = vx if vx is not None else (- Vmax * (abs(self.y - 185) / 185 ) ** 2 + Vmax - Vz * (abs(self.z) / 150) ** 2 + Vz)


    def update(self):
        self.x = (self.x + self.vx) % W
        self.y = (self.y + self.vy) % H



# Sample points
point_groups = []
for _ in range(num_groups):
    vx = np.random.randint(-10, 10)
    vy = np.random.randint(-10, 10)

    points = [Point(radius) for i in range(num_points)]
    point_groups.append(points)



def drawPoint(img, point: Point):
    half_size = point.size
    size = 2 * half_size + 1
    w, h = img.shape
    x = point.x # int(point.x + .5) # int(np.clip(point.x, size, w - size) + .5)
    y = point.y # int(point.y + .5) # int(np.clip(point.y, size, h - size) + .5)

    #draw particles in square shape
#   # region = slice(x - half_size, x + half_size + 1), slice(y - half_size, y + half_size + 1)
    # img[region] += point.intensity
    # img[region] = np.clip(img[region], 0, 255)
    # print(region)

    #draw particles in blob
    img += point.intensity * (np.sqrt(2) / half_size) * np.exp(-(((x - pixX.T) / half_size)**2 + ((y - pixY.T) / half_size)**2))
    img = np.clip(img, 0, 255)

# Update and draw points
for points in point_groups:
    for p in points:
        drawPoint(img1, p)
        p.update()
        drawPoint(img2, p)

file1 = Image.fromarray(img1.T.astype(np.uint8))
file2 = Image.fromarray(img2.T.astype(np.uint8))

file1.save('./image1.bmp')
file2.save('./image2.bmp')
