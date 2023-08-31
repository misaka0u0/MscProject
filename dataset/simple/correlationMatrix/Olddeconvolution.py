import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import tifffile
import os

import correlationMatrix
from skimage import color, data, restoration

output_dir = "./deconvolved_Matrix"

os.makedirs(output_dir, exist_ok=True)


def main():
    # Load images
    psf = np.load("./PSF.npy")  # (D, H, W)
    objective = np.load("./corr_matrix.npy")  # (D, grid_y, grid_x, H, W)

    # trnaspose for process by index of grid_y, grid_x
    objective = objective.transpose((1, 2, 0, 3, 4))

    for grid_coord in np.ndindex(objective[:-3]):

        corr_matrix = objective[grid_coord]

        deconvolved_matrix = restoration.richardson_lucy(corr_matrix, psf, num_iter=250,clip=False)




        pixel_center = kernel_centers[grid_coord]

        # Set up windows
        interrogation_window = current_frame[
            create_selection_from_center(center=pixel_center, shape=interrogation_window_shape)
        ]

        padded_next_frame = zero_padding(arr=next_frame, target_shape=next_frame.shape + search_window_shape)
        padding_offset: ndarray2 = get_zero_padding_offset(
            original_shape=next_frame.shape, target_shape=next_frame.shape + search_window_shape
        )
        search_window = padded_next_frame[
            create_selection_from_center(center=pixel_center + padding_offset, shape=search_window_shape)
        ]

        # Compute the cross-correlations
        cross_corr = correlate(
            in1=search_window,
            in2=interrogation_window,
            mode="valid",
            method="fft",
        )

        # Store the matrix in the grid
        correlation_results[grid_coord] = cross_corr

        offset = np.unravel_index(np.argmax(cross_corr), cross_corr.shape)  # TODO: This control is weak.
        offset -= correlation_patch_result_size // 2
        correlation_maximum_offsets[grid_coord] = offset




if __name__ == "__main__":
    main()




int_win_size = np.array([32, 32])
search_win_size = np.array([64, 64])
half_int_win_size = int_win_size // 2
half_search_win_size = search_win_size // 2
corr_win_size = search_win_size - int_win_size + 1
half_corr_win_size = (corr_win_size - 1) // 2 + 1

# plt.imshow(np.max(img1, axis=1), aspect=8.0)


# Restore Image using Richardson-Lucy algorithm
deconvolved_Matrix = restoration.richardson_lucy(corrMatrix, psf, num_iter=50,clip=False)



# plt.imshow(deconvolved_RL, cmap = plt.cm.gray)
# plt.show()
# 2d only

import os
# Ensure the directory for the images exists

# Normalize the image data to 0-255
deconvolved_Matrix = ((deconvolved_Matrix - deconvolved_Matrix.min()) * (1/(deconvolved_Matrix.max() - deconvolved_Matrix.min()) * 255)).astype('uint8')

np.save('deconvolved_corrM.npy', deconvolved_Matrix)
# data = np.load('filename.npy')

velocity_stack = []
velocity_stack_2d = []
# Iterate over the image stack
for i in range(deconvolved_Matrix.shape[0]):
    # Convert numpy array to PIL image
    img = deconvolved_Matrix[i]

    # Save the image
    # Normalize to [0, 1]
    normalized_array = (img - img.min()) / (img.max() - img.min())

    # Save as a 32-bit floating point TIFF image
    Image.fromarray(normalized_array.T.astype(np.float32)).save(f'./deconvolved_Matrix/image_{i * 10}.tif')


    # total_size_y = corrMatrix.shape[0] // int_win_size[0] * corr_win_size[0]
    # total_size_x = corrMatrix.shape[1] // int_win_size[1] * corr_win_size[1]
    # # initialize 
    # correlation_array = np.zeros((total_size_y, total_size_x))
    
    ys = np.arange(half_corr_win_size[0], corrMatrix.shape[0], 2 * half_corr_win_size[0])
    xs = np.arange(half_corr_win_size[1], corrMatrix.shape[1], 2 * half_corr_win_size[1])
    dys = np.zeros((len(ys), len(xs)))
    dxs = np.zeros((len(ys), len(xs)))
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            
            start_y = iy * corr_win_size[0]
            end_y = start_y + corr_win_size[0]
            start_x = ix * corr_win_size[1]
            end_x = start_x + corr_win_size[1]
            if end_y <= img.shape[0] and end_x <= img.shape[1]:
                dy, dx = (np.unravel_index(np.argmax(img[start_y: end_y, start_x: end_x]), corr_win_size) 
                    - half_corr_win_size)

            dys[iy, ix] = dy
            
    # print(dys)              # []        (xxxx)
    # print(dys.shape)        # (0, 3)    (0, 10)
    # print(img.shape[0])     # 330       330
    # print(img.shape[1])     # 330       330
    # print(corr_win_size[0]) # 129       33
    # print(corr_win_size[1]) # 129       33

    v0 = np.average(dys) # 0d s
    velocity_stack.append(v0)
    # velocity_stack_2d.append(dys) # 2d s
    velocity_stack_2d.append(np.max(dys, axis=0)) # 1d s

x = range(corrMatrix.shape[0])

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
plt.show()

# #  ============ Heat Map =================

# # Convert to a numpy array for convenience
# velocity_stack_2d = np.array(velocity_stack_2d)

# # Now you can create a heatmap. Note that we use the `imshow` function
# # and also include a colorbar.
# plt.figure(figsize=(10, 8))
# plt.imshow(velocity_stack_2d, aspect='auto', cmap='hot', origin='lower')
# plt.colorbar(label='Velocity')
# plt.xlabel('Y position')
# plt.ylabel('Z position')
# plt.title('Velocity Heatmap')
# plt.show()

#  ============= first correlation Map=============
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
Y, X = np.meshgrid(np.arange(deconvolved_Matrix.shape[1]), np.arange(deconvolved_Matrix.shape[2]))

ax.plot_surface(Y, X, deconvolved_Matrix[0].T, cmap='jet', linewidth=0.2)  # type: ignore
plt.title("Correlation map — peak is the most probable shift")
plt.show()


sidedec = np.max(deconvolved_Matrix[:,:,10:40], axis=2)
tifffile.imwrite('sidedec.tif', sidedec)