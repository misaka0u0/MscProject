import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from PIL import Image

from scipy.signal import convolve2d as conv2

from skimage import color, data, restoration


psf = np.load('PSF.npy')
img1 = np.load('stkA.npy')
img2 = np.load('stkB.npy')


# Restore Image using Richardson-Lucy algorithm
deconvolved_RL1 = restoration.richardson_lucy(img1, psf, num_iter=150,clip=False)
deconvolved_RL2 = restoration.richardson_lucy(img2, psf, num_iter=150,clip=False)

# plt.imshow(deconvolved_RL, cmap = plt.cm.gray)
# plt.show()
# 2d only

import os
# Ensure the directory for the images exists
os.makedirs('./deconvolved_images1', exist_ok=True)
os.makedirs('./deconvolved_images2', exist_ok=True)

# Normalize the image data to 0-255
deconvolved_RL1 = ((deconvolved_RL1 - deconvolved_RL1.min()) * (1/(deconvolved_RL1.max() - deconvolved_RL1.min()) * 255)).astype('uint8')
deconvolved_RL2 = ((deconvolved_RL2 - deconvolved_RL2.min()) * (1/(deconvolved_RL2.max() - deconvolved_RL2.min()) * 255)).astype('uint8')

# Iterate over the image stack
for i in range(deconvolved_RL1.shape[0]):
    # Convert numpy array to PIL image
    img1 = deconvolved_RL1[i]

    # Save the image
    Image.fromarray(img1.T.astype(np.uint8)).save(f'./deconvolved_images1/image1_{i * 10}.bmp')

for i in range(deconvolved_RL2.shape[0]):
    # Convert numpy array to PIL image
    img2 = deconvolved_RL2[i]

    # Save the image
    Image.fromarray(img2.T.astype(np.uint8)).save(f'./deconvolved_images2/image2_{i * 10}.bmp')

np.save('deconvolved_RL1.npy', deconvolved_RL1)
np.save('deconvolved_RL2.npy', deconvolved_RL2)