import matplotlib.pyplot as plt
import numpy as np
import tifffile
import os
from PIL import Image

import correlationMatrix as corrM
import edited_RL as eRL
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
    print(f'corr_max_offsets.shape:', correlation_maximum_offsets.shape)

    # trnaspose for process by index of grid_y, grid_x
    objective = objective.transpose((1, 2, 0, 3, 4)) # [D, grid_y, grid_x, [H, W]] -> [grid_y, grid_x, D, [H, W]]

    for grid_coord in np.ndindex(objective.shape[:-3]):

        corr_matrix = objective[grid_coord]
        print(f'corr_matrix', corr_matrix.shape)
        # deconvolved_matrixes = restoration.richardson_lucy(corr_matrix, psf, num_iter=50,clip=False)
        deconvolved_matrixes = eRL.richardson_lucy_edge(corr_matrix, psf, num_iter=150,clip=False)
        deconvolved_results[grid_coord] = deconvolved_matrixes
        
    deconvolved_results = deconvolved_results.transpose((2, 0, 1, 3, 4))
    print(f'deconvolved_results shape:', deconvolved_results.shape)


    for depth, deconvolved_matrixes in enumerate(deconvolved_results):

        normalized_deconvolved_matrix =  corrM.normalize(deconvolved_matrixes)
        output_2d_layout = normalized_deconvolved_matrix.transpose((0, 2, 1, 3)).reshape((h * y, w * x))
        print(f'deconvolved_matrixes', deconvolved_matrixes.shape)

        os.makedirs(output_dir, exist_ok=True)
        Image.fromarray(output_2d_layout.__mul__(255).astype(np.uint8), mode="L").save(
            os.path.join(output_dir, f"dz_{depth * 10}.tif")
        )
        edge = 3 # the edge value
        for grid_coord in np.ndindex(objective.shape[:-3]):
    
            # deconvolved_matrix = deconvolved_results[depth, grid_coord]
            deconvolved_matrix = deconvolved_matrixes[grid_coord]
            deconvolved_matrix = deconvolved_matrix[edge: y-edge, edge: x-edge] # edge to be cut
            print(f'grid_', grid_coord)
            print(f'deconvolved_matrix shape in for:', deconvolved_matrix.shape)
            offset = np.unravel_index(np.argmax(deconvolved_matrix), deconvolved_matrix.shape)
            offset = offset - ( np.array((y-2*edge, x-2*edge), dtype=int) // 2)
            offset = np.stack(offset)
            print(f'offset shape:', offset.shape)
            correlation_maximum_offsets[grid_coord] = offset

        v_x = np.mean(correlation_maximum_offsets[:, :, 1])
        velocity_x_stack.append(v_x)
        velocity_stack.append(correlation_maximum_offsets)

    corrM.plot_velocity_curve(velocity_x_stack)
    # corrM.plot_zox_view(deconvolved_results)
    # corrM.plot_zoy_view(velocity_stack)
    # corrM.plot_surface(output_2d_layout)
    plt.show()

if __name__ == "__main__":
    main()