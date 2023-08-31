import matplotlib.pyplot as plt
import numpy as np
import tifffile
import os
from PIL import Image

import correlationMatrix
from skimage import color, data, restoration

output_dir = "./deconvolved_Matrix"

os.makedirs(output_dir, exist_ok=True)


def main():
    # Load images
    psf = np.load("./PSF.npy")  # (D, H, W)
    objective = np.load("./corr_matrix.npy")  # (D, grid_y, grid_x, H, W)
    d, h, w, y, x = objective.shape
    print(f'The shape of objective is:', objective.shape)

    velocity_stack = []
    velocity_x_stack = []
    correlation_maximum_offsets = np.zeros([objective.shape[1], objective.shape[2], 2], dtype=int)
    deconvolved_results = np.zeros(np.concatenate(([h, w, d], [y, x])))
    print(deconvolved_results.shape)

    # trnaspose for process by index of grid_y, grid_x
    objective = objective.transpose((1, 2, 0, 3, 4)) # [D, grid_y, grid_x, [H, W]] -> [grid_y, grid_x, D, [H, W]]

    for grid_coord in np.ndindex(objective.shape[:-3]):

        corr_matrix = objective[grid_coord]
        deconvolved_matrix = restoration.richardson_lucy(corr_matrix, psf, num_iter=250,clip=False)
        deconvolved_results[grid_coord] = deconvolved_matrix
        
    deconvolved_results = deconvolved_results.transpose((2, 0, 1, 3, 4))


    for depth, deconvolved_matrix in enumerate(deconvolved_results[ :, :, D, :, :] for D in range(deconvolved_results.shape[2])):

        normalized_deconvolved_matrix = normalize(deconvolved_matrix)
        output_2d_layout = normalized_deconvolved_matrix.transpose((0, 2, 1, 3)).reshape((h * y, w * x))

        os.makedirs(output_dir, exist_ok=True)
        Image.fromarray(output_2d_layout.__mul__(255).astype(np.uint8), mode="L").save(
            os.path.join(output_dir, f"dz_{depth * 10}.tif")
        )

        for grid_coord in np.ndindex(objective[:-3]):
    
            deconvolved_matrix = deconvolved_results[grid_coord]
            offset = np.unravel_index(np.argmax(deconvolved_matrix), deconvolved_matrix.shape)  
            correlation_maximum_offsets[grid_coord] = offset

        v_x = np.mean(correlation_maximum_offsets[:, :, 1])
        velocity_x_stack.append(v_x)
        velocity_stack.append(correlation_maximum_offsets)

    plot_velocity_curve(velocity_x_stack)
    # plot_zox_view(correlation_2d_layout_stack)
    # plot_zoy_view(velocity_stack)
    # plot_surface(output_2d_layout)
    plt.show()

if __name__ == "__main__":
    main()