import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


width, height = 511, 369
radius = 2
num_points = 1000
intensity = 125

np.random.seed(42)

# initialise the image
img1 = np.zeros([height, width], dtype=np.uint8)
img2 = np.zeros([height, width], dtype=np.uint8)

xs = np.random.choice(range(radius, width - radius), size=[num_points, 1])
ys = np.random.choice(range(radius, height - radius), size=[num_points, 1])
points1 = np.hstack([ys, xs])

xs = np.random.choice(range(radius, width - radius), size=[num_points, 1])
ys = np.random.choice(range(radius, height - radius), size=[num_points, 1])
points2 = np.hstack([ys, xs])

# plot the particle with velovity v1 for particle group1 and v2 for group 2
# v1 = np.random.randint(-3, 3, dtype=np.int8, size=(1, 2))
# v2 = np.random.randint(-3, 3, dtype=np.int8, size=(1, 2))
v1 = [ 2, 4]
v2 = [-4, -4]
# print(v1, v2)


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

file1 = Image.fromarray(img1)
file2 = Image.fromarray(img2)

file1.save('./image1.bmp')
file2.save('./image2.bmp')


# Create a new image to combine the colors
combined_image = np.zeros([height, width, 3], dtype=np.uint8)

# Set the red color for particles from image1
combined_image[img1 > 0, 0] = 255

# Set the blue color for particles from image2
combined_image[img2 > 0, 2] = 255

# Save the combined image
combined_file = Image.fromarray(combined_image)
combined_file.save('./combined_image.bmp')
