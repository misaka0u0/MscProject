import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from PIL import Image

from scipy.signal import convolve

from skimage import color, data, restoration


psf = np.load('PSF.npy')
point = np.load('point.npy')

img = convolve(point, psf,mode='same')

import os
# Ensure the directory for the images exists
os.makedirs('./result', exist_ok=True)


# Iterate over the image stack
for i in range(img.shape[0]):
    # Convert numpy array to PIL image
    image = img[i]

    # Save the image
    Image.fromarray(image.astype(np.uint8)).save(f'./result/image_{i}.bmp')

np.save('object.npy', img)