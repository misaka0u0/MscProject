import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

width, height = 511, 369
radius = 2
num_points = 300
intensity = 125

# initialise the image
img1 = np.zeros([width, height], dtype=np.uint8)
img2 = np.zeros([width, height], dtype=np.uint8)

xs = np.random.choice(range(radius, width - radius), size=[num_points, 1])
ys = np.random.choice(range(radius, height - radius), size=[num_points, 1])
points1 = np.hstack([xs, ys])

xs = np.random.choice(range(radius, width - radius), size=[num_points, 1])
ys = np.random.choice(range(radius, height - radius), size=[num_points, 1])
points2 = np.hstack([xs, ys])

# plot the particle with velovity v1 for particle group1 and v2 for group 2
v1 = np.random.randint(-10, 10, dtype=np.int8, size=(1, 2))
v2 = np.random.randint(-10, 10, dtype=np.int8, size=(1, 2))
print(v1, v2)

def drawPoint(img, x, y, half_size, intensity):
    size = 2 * half_size + 1
    w, h = img.shape
    region = slice(x - half_size, x + half_size + 1), slice(y - half_size, y + half_size + 1)
    img[region] += intensity
    img[region] = np.clip(img[region], 0, 255)


for points_group in [points1, points2]:
    for x, y in points_group:
        drawPoint(img1, x, y, half_size=radius, intensity=intensity)

# Movement
points1 += v1
points2 += v2

for points_group in [points1, points2]:
    for x, y in points_group:
        drawPoint(img2, x, y, half_size=radius, intensity=intensity)

file1 = Image.fromarray(img1.T)
file2 = Image.fromarray(img2.T)

file1.save('./particles1.bmp')
file2.save('./particles2.bmp')
