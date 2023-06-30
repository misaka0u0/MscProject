import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#size all in pixels
# Specify image size
imageWidth = 511
imageHeight = 369

# Set the velocity vector for all particles
# velocityX = 2  
# velocityY = 4  

# Generate an initial image 
imageData1 = np.zeros((imageWidth, imageHeight), dtype=np.uint8)
imageData2 = np.zeros((imageWidth, imageHeight), dtype=np.uint8)

#Generate randomly placed particles, and include parameters
numPaticles = 20000
particleRadius = 2
particleIntensity = 125 

xs = np.random.choice(range(particleRadius, imageWidth - particleRadius ), [numPaticles, 1])
ys = np.random.choice(range(particleRadius, imageHeight - particleRadius ), [numPaticles, 1])
particlesGroup1 = np.concatenate([xs, ys], axis = 1)

velocityX = np.random.choice(range(particleRadius, imageWidth - particleRadius ), [numPaticles, 1])
velocityY = np.random.choice(range(particleRadius, imageHeight - particleRadius ), [numPaticles, 1])

for position in particlesGroup1:
    yCenter = position[1]
    xCenter = position[0]
    imageData1[xCenter ,
               yCenter ] += particleIntensity
    imageData1 = np.clip(imageData1, 0, 255)
    # imageData1[xCenter, yCenter] += 130
    # imageData1 = np.clip(imageData1, 0, 255)

for position in particlesGroup1:
    x, y =  np.array([velocityX, velocityY])
    imageData2[x , y ] += particleIntensity

# Transpose for pillow  
image1 = Image.fromarray(imageData1.T)
image2 = Image.fromarray(imageData2.T)

# Save the images in .bmp format
image1.save('./image1.bmp')
image2.save('./image2.bmp')
