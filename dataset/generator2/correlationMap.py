import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread

# load images
image1 = imread("./image1.bmp")
image2 = imread("./image2.bmp")

win_size = 32

a_win = image1 [:win_size, :win_size].copy()
b_win = image2 [:win_size, :win_size].copy()

# fig, axs = plt.subplots(1, 2, figsize=(9, 4))
# axs[0].imshow(a_win, cmap=plt.cm.gray)
# axs[1].imshow(b_win, cmap=plt.cm.gray)
# plt.show()

# def match_template(img, template, maxroll=8):
#     best_dist = np.inf
#     best_shift = (-1, -1)
#     for y in range(maxroll):
#         for x in range(maxroll):
#             # calculate Euclidean distance
#             dist = np.sqrt(np.sum((img - np.roll(template, (y, x), axis=(0, 1))) ** 2))
#             if dist < best_dist:
#                 best_dist = dist
#                 best_shift = (y, x)
#     return (best_dist, best_shift)

# best_dist, best_shift = match_template(np.roll(a_win, (2, 0), axis=(0, 1)), a_win)
# print(best_dist, best_shift)

# best_dist, best_shift = match_template(b_win, a_win)
# print(f"{best_dist=}")
# print(f"{best_shift=}")

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