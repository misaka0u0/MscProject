import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread

# load images
image1 = imread("./image1.bmp")
image2 = imread("./image2.bmp")

win_size = 128

a_win = image1 [:win_size, :win_size].copy()
b_win = image2 [:win_size, :win_size].copy()

from scipy.signal import correlate

cross_corr = correlate(b_win - b_win.mean(), a_win - a_win.mean(), method="fft")

# print("%d x %d" % cross_corr.shape)

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
Y, X = np.meshgrid(np.arange(cross_corr.shape[0]), np.arange(cross_corr.shape[1]))

ax.plot_surface(Y, X, cross_corr, cmap='jet', linewidth=0.2)  # type: ignore
plt.title("Correlation map — peak is the most probable shift")
plt.show()