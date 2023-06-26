import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#size all in pixels
# Specify image size
imageWidth = 511
imageHeight = 369

# Set the velocity vector for all particles
velocityX = 2  
velocityY = 4  

# Generate an initial image 
imageData1 = np.zeros((imageWidth, imageHeight), dtype=np.uint8)
imageData2 = np.zeros((imageWidth, imageHeight), dtype=np.uint8)

#Generate randomly placed particles, and include parameters
numPaticles = 80
particleRadius = 2
particleIntensity = 125 

xs = np.random.choice(range(particleRadius, imageWidth - particleRadius), [numPaticles, 1])
ys = np.random.choice(range(particleRadius, imageHeight - particleRadius), [numPaticles, 1])
particlesGroup1 = np.concatenate([xs, ys], axis = 1)

# print(xs)
                              #                  2                      511 - 2              (500,2)
# particlesGroup1 = np.random.randint(particleRadius, imageWidth - particleRadius, size = (number, 2))
# particlesGroup1[:, 1] = np.clip(particlesGroup1[:, 1], particleRadius, imageHeight - particleRadius -1)
# print(min(particlesGroup1.reshape(-1)))
# print(max(particlesGroup1.reshape(-1)))

# plt.scatter(particlesGroup1[:, 0], particlesGroup1[:, 1])
# plt.show()
#random number should generate from at least particle radius to avoid extending
#np.clip to avoid exceeding 
#print(particlesGroup1)


for position in particlesGroup1:
    yCenter = position[1]
    xCenter = position[0]
    imageData1[xCenter - particleRadius: xCenter + particleRadius + 1,
               yCenter - particleRadius: yCenter + particleRadius + 1] += particleIntensity
    imageData1 = np.clip(imageData1, 0, 255)
    # imageData1[xCenter, yCenter] += 130
    # imageData1 = np.clip(imageData1, 0, 255)

for position in particlesGroup1:
    x, y = position +  np.array([velocityX, velocityY])
    imageData2[x - particleRadius: x + particleRadius + 1,
               y - particleRadius: y + particleRadius + 1] += particleIntensity
# image_data_1[50:100, 50:100] = 255  # Add particles at specific region
# image_1 = Image.fromarray(image_data_1)


# image_data_2 = np.zeros((image_height, image_width), dtype=np.uint8)
# for y in range(image_height):
#     for x in range(image_width):
#         if image_data_1[y, x] == 255:
#             new_x = int(x + velocity_x)
#             new_y = int(y + velocity_y)
#             if new_x >= 0 and new_x < image_width and new_y >= 0 and new_y < image_height:
#                 image_data_2[new_y, new_x] = 255

# Transpose for pillow  
image1 = Image.fromarray(imageData1.T)
image2 = Image.fromarray(imageData2.T)

# Save the images in .bmp format
image1.save('./image1.bmp')
image2.save('./image2.bmp')
