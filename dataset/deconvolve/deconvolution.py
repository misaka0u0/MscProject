import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from PIL import Image

from scipy.signal import convolve2d as conv2

from skimage import color, data, restoration


psf = np.load('PSF.npy')
# img = np.load('stk3d.npy')
img = np.load('object.npy')


# Restore Image using Richardson-Lucy algorithm
deconvolved_RL = restoration.richardson_lucy(img, psf, num_iter=5)

# plt.imshow(deconvolved_RL, cmap = plt.cm.gray)
# plt.show()
# 2d only

import os
# Ensure the directory for the images exists
os.makedirs('./deconvolved_images', exist_ok=True)

# Normalize the image data to 0-255
deconvolved_RL = ((deconvolved_RL - deconvolved_RL.min()) * (1/(deconvolved_RL.max() - deconvolved_RL.min()) * 255)).astype('uint8')

# Iterate over the image stack
for i in range(deconvolved_RL.shape[0]):
    # Convert numpy array to PIL image
    img = deconvolved_RL[i]

    # Save the image
    Image.fromarray(img.T.astype(np.uint8)).save(f'./deconvolved_point/image_{i}.bmp')