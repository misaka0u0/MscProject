import matplotlib.pyplot as plt
from enum import IntEnum
from operator import length_hint
from matplotlib import image
from PIL import Image
from typing import Optional
import numpy as np

from scipy.signal import correlate

# load images
objectA = np.load('stkA.npy')
objectB = np.load('stkB.npy')

win_size = 32

import os
# Check if the directories exist, and if not, create them
os.makedirs('./CorrelationMatrix', exist_ok=True)

# correlation_stack = []

# for i in range(objectA.shape[0]):  # Iterate over possible dz values
    
#     img1 = objectA[i]
#     img2 = objectB[i]

#     a_win = img1 [:win_size, :win_size].copy()
#     b_win = img2 [:win_size, :win_size].copy()

#     cross_corr = correlate(b_win - b_win.mean(), a_win - a_win.mean(), method="fft")

#     correlation_stack.append(cross_corr)

#     # Save the images
#     Image.fromarray(cross_corr.T.astype(np.uint8)).save(f'./CorrelationMatrix/dz_{i * 10}.bmp')



# # load images
# objectA = np.load('stkA.npy')
# objectB = np.load('stkB.npy')

# int_win_size = 32
# search_win_size = 36 

# img1 = objectA[0]
# img2 = objectB[0]

# a_win = img1[:int_win_size, :int_win_size].copy()
# b_win = img2[:search_win_size, :search_win_size].copy()

# cross_corr = correlate(b_win - b_win.mean(), a_win - a_win.mean(), mode="same", method="fft")
# print(cross_corr.shape)
# # cross_corr shape in (36, 36)


# np.save('corrMatrix.npy', correlation_stack)
# data = np.load('filename.npy')

# from mpl_toolkits.mplot3d import Axes3D

# fig = plt.figure()
# ax = fig.add_subplot(projection="3d")
# Y, X = np.meshgrid(np.arange(cross_corr.shape[0]), np.arange(cross_corr.shape[1]))

# ax.plot_surface(Y, X, correlation_stack[0], cmap='jet', linewidth=0.2)  # type: ignore
# plt.title("Correlation map — peak is the most probable shift")
# plt.show()



int_win_size = np.array([32, 32])
search_win_size = np.array([64, 64])#36 - 32 = 4
half_int_win_size = int_win_size // 2
half_search_win_size = search_win_size // 2
corr_win_size = search_win_size - int_win_size + 1

def correlate_and_combine(
    img1, img2, half_int_win_size, half_search_win_size
):

    total_size_y = img1.shape[0] // int_win_size[0] * corr_win_size[0]
    total_size_x = img1.shape[1] // int_win_size[1] * corr_win_size[1]
    # initialize 
    correlation_array = np.zeros((total_size_y, total_size_x))
    
    ys = np.arange(half_int_win_size[0], img1.shape[0], 2 * half_int_win_size[0])
    xs = np.arange(half_int_win_size[1], img1.shape[1], 2 * half_int_win_size[1])
    dys = np.zeros((len(ys), len(xs)))
    dxs = np.zeros((len(ys), len(xs)))

    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            int_win = img1[
                y - half_int_win_size[0] : y + half_int_win_size[0],
                x - half_int_win_size[1] : x + half_int_win_size[1],
            ]
            search_win_y_min = y - half_search_win_size[0]
            search_win_y_max = y + half_search_win_size[0]
            search_win_x_min = x - half_search_win_size[1]
            search_win_x_max = x + half_search_win_size[1]
            truncated_search_win = img2[
                max(0, search_win_y_min) : min(img2.shape[0], search_win_y_max),
                max(0, search_win_x_min) : min(img2.shape[1], search_win_x_max),
            ]
            # cross_corr = correlate(
            #     truncated_search_win - np.mean(truncated_search_win),
            #     int_win - np.mean(int_win),
            #     mode="valid",
            #     method="fft",
            # )
            cross_corr = correlate(
                truncated_search_win ,
                int_win ,
                mode="valid",
                method="fft",
            )
            
            start_y = iy * cross_corr.shape[0]
            end_y = start_y + cross_corr.shape[0]
            start_x = ix * cross_corr.shape[1]
            end_x = start_x + cross_corr.shape[1]
    
            # place the result in correct location
            correlation_array[start_y: end_y, start_x: end_x] = cross_corr

            dys[iy, ix], dxs[iy, ix] = (
                np.unravel_index(np.argmax(cross_corr), cross_corr.shape) 
                # - np.array([win_size, win_size]) + 1
            )

    return correlation_array, dxs, dys


correlation_stack = []
velocity_stack =[]

for i in range(objectA.shape[0]):  # Iterate over possible dz values
    
    img1 = objectA[i]
    img2 = objectB[i]

    correlation_array,dxs, dys = correlate_and_combine(
        img1=img1, 
        img2=img2, 
        half_int_win_size=half_int_win_size, 
        half_search_win_size=half_search_win_size
    )

    correlation_stack.append(correlation_array)

    # Save the images
    # Normalize to [0, 1]
    normalized_array = (correlation_array - correlation_array.min()) / (correlation_array.max() - correlation_array.min())

    # Save as a 32-bit floating point TIFF image
    Image.fromarray(normalized_array.T.astype(np.float32)).save(f'./CorrelationMatrix/dz_{i * 10}.tif')

# print(dys)
# print(dxs)

# # ------------HEATMAP-----------------

#   velocity_stack.append(np.max(v0, axis=0))  # Using the maximum velocity value along y-axis for each layer

# # Convert to a numpy array for convenience
# velocity_stack = np.array(velocity_stack)

# # Now you can create a heatmap. Note that we use the `imshow` function
# # and also include a colorbar.
# plt.figure(figsize=(10, 8))
# plt.imshow(velocity_stack, aspect='auto', cmap='hot', origin='lower')
# plt.colorbar(label='Velocity')
# plt.xlabel('Y position')
# plt.ylabel('Z position')
# plt.title('Velocity Heatmap')
# plt.show()


# -----------Velocity profile--------------
    v0 = np.average(dxs)
    # v0 = np.max(v0)
    velocity_stack.append(v0)

print(velocity_stack)

x = range(objectA.shape[0])

y = velocity_stack
fig = plt.figure()
ax  = fig.add_subplot(111)

# Fit a parabola to the data
coeffs = np.polyfit(x, y, 2)  # Fit a 2nd degree polynomial
poly = np.poly1d(coeffs)  # Create a polynomial function
yfit = poly(x)  # Generate fitted y-values

# Plot
plt.scatter(x, y, label='velocity profile')
plt.plot(x, yfit, color='red', label='fitted curve')  # Plot the fitted curve

# plt.plot(x, y, label='velocity profile')
plt.xlabel('z')
plt.ylabel('velocity')
plt.title('velocity profile')
plt.legend()
plt.grid(True)
plt.ylim([-4, 12])
plt.show()





print(correlation_array.shape)

correlation_stack = np.stack(correlation_stack)
np.save('corrMatrix.npy', correlation_stack)
# data = np.load('filename.npy')

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
Y, X = np.meshgrid(np.arange(correlation_array.shape[0]), np.arange(correlation_array.shape[1]))

ax.plot_surface(Y, X, correlation_stack[0].T, cmap='jet', linewidth=0.2)  # type: ignore
plt.title("Correlation map — peak is the most probable shift")
plt.show()
