import numpy as np
import matplotlib.pyplot as plt

height = 150
width = 200

# Generate an initially black image
image = np.zeros((height, width))

# Generate two arrays the same shape as the image.
# In pixX, every array element has a value equal to the x coordinate of that pixel
# In pixY, every array element has a value equal to the y coordinate of that pixel
pixX, pixY = np.meshgrid(np.arange(width), np.arange(height))

def DrawParticle(image, x, y, peak_intensity, particle_radius):
    # This function draws a particle onto the image (at coordinate x,y).
    # The equation describing the intensity as a function of position relative to the particle coordinate
    # (relative position x-pixX, y-pixY) is a Gaussian function.
    image += peak_intensity * (np.sqrt(2) / particle_radius) * np.exp(-(((x-pixX)/particle_radius)**2 + ((y-pixY)/particle_radius)**2))

# Draw a single example particle into the image
DrawParticle(image, 50, 50, 1, 2.0)
# Display the image on screen
plt.imshow(image, cmap='gray')
plt.show()
# print(pixX)
# print (((50-pixX)/5.0)**2)
# print(image[50, 50])