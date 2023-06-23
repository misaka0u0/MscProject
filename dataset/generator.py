import numpy as np
from PIL import Image

# Specify image size
image_width = 511
image_height = 369

# Set the velocity vector for all particles
velocity_x = 225  
velocity_y = 150  

# Generate an blanket image 
image_data_1 = np.zeros((image_height, image_width), dtype=np.uint8)
image_data_1[50:100, 50:100] = 255  # Add particles at specific region
image_1 = Image.fromarray(image_data_1)


image_data_2 = np.zeros((image_height, image_width), dtype=np.uint8)
for y in range(image_height):
    for x in range(image_width):
        if image_data_1[y, x] == 255:
            new_x = int(x + velocity_x)
            new_y = int(y + velocity_y)
            if new_x >= 0 and new_x < image_width and new_y >= 0 and new_y < image_height:
                image_data_2[new_y, new_x] = 255
image_2 = Image.fromarray(image_data_2)

# Save the images in .bmp format
image_1.save('image1.bmp')
image_2.save('image2.bmp')
