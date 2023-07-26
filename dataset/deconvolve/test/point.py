# import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from typing import Optional

W, H = 511, 369 # x, y axis
Length = 150 # z axis

intensity = 125 # The max intensity, 'a' in Gaussian blob


# Generate two arrays the same shape as the image.
# In pixX, every array element has a value equal to the x coordinate of that pixel
# In pixY, every array element has a value equal to the y coordinate of that pixel
pixX, pixY = np.meshgrid(np.arange(W), np.arange(H))

class Point:
    def __init__(self, dz: Optional[int] = None):
        self.x = 260
        self.y = 180
        self.z = 0
        self.dz = dz if dz is not None else 0
        self.intensity = intensity 

def drawPoint(img, point: Point):
    img[point.y, point.x] = point.intensity
    img = np.clip(img, 0, 255)


import os

# Check if the directories exist, and if not, create them
os.makedirs('./point', exist_ok=True)

image_stack = []
for dz in range(-Length, Length + 1, 10):  # Iterate over possible dz values
    np.random.seed(42)
    points = [Point(dz=dz) for _ in range(1)]  # Initialize points with the current dz
    # Initialize the images
    img1 = np.zeros([H, W])

    # Draw point only when dz = 0
    for p in points:
        if p.dz == 0:
            drawPoint(img1, p)

    image_stack.append(img1)
    Image.fromarray(img1.astype(np.uint8)).save(f'./point/dz_{dz+150}.bmp')

# Convert the list of 2D arrays into a 3D numpy array
image_stack_3d = np.stack(image_stack)

# print(image_stack1)
print(np.shape(image_stack))
print(np.shape(image_stack_3d))

np.save('point.npy', image_stack_3d)
# data = np.load('filename.npy')

