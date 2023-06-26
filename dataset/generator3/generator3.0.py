# import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from typing import Optional

num_groups = 1
num_points = 80
W, H = 511, 369
radius = 2
Vmax = 35

img1 = np.zeros([W, H], dtype=np.uint8)
img2 = np.zeros([W, H], dtype=np.uint8)

class Point:
    def __init__(self, size: int, vx: Optional[int] = None, vy: Optional[int] = None):
        self.x = np.random.randint(size, W - size)
        self.y = np.random.randint(size, H - size)
        self.vx = vx if vx is not None else (- Vmax * (abs(self.y - 185) / 185 ) ** 2 + Vmax)
        self.vy = vy if vy is not None else 0
        self.size = size
        self.intensity = 125


    def update(self):
        self.x += self.vx
        self.y += self.vy


# Sample points
point_groups = []
for _ in range(num_groups):
    # vx = np.random.randint(-10, 10)
    # vy = np.random.randint(-10, 10)

    points = [Point(2) for i in range(num_points)]
    point_groups.append(points)



def drawPoint(img, point: Point):
    half_size = point.size
    size = 2 * half_size + 1
    w, h = img.shape
    x = int(np.clip(point.x, size, w - size) + .5)
    y = int(np.clip(point.y, size, h - size) + .5)
    region = slice(x - half_size, x + half_size + 1), slice(y - half_size, y + half_size + 1)
    img[region] += point.intensity
    img[region] = np.clip(img[region], 0, 255)
    # print(region)

# Update and draw points
for points in point_groups:
    for p in points:
        drawPoint(img1, p)
        p.update()
        drawPoint(img2, p)

file1 = Image.fromarray(img1.T)
file2 = Image.fromarray(img2.T)

file1.save('./image1.bmp')
file2.save('./image2.bmp')
