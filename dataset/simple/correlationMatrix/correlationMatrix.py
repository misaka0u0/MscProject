import os
from typing import TypeAlias

import matplotlib.pyplot as plt
import numpy as np
import PIL
from alive_progress import alive_it
from numpy import float32, ndarray
from scipy.signal import correlate
from mpl_toolkits.mplot3d import Axes3D

output_dir = "./CorrelationMatrix"

ndarray2: TypeAlias = ndarray
ndarray3: TypeAlias = ndarray
ndarray4: TypeAlias = ndarray
ndarray5: TypeAlias = ndarray


def get_zero_padding_offset(original_shape: ndarray | tuple[int], target_shape: ndarray | tuple[int]) -> ndarray2:
    shape_from = np.array(original_shape, dtype=int)
    shape_to = np.array(target_shape, dtype=int)
    offset = shape_to - shape_from
    offset //= 2
    return offset


def zero_padding(arr: ndarray, target_shape: ndarray | tuple[int]) -> ndarray:
    target_shape = np.array(target_shape, dtype=int)
    buffer = np.zeros(target_shape, dtype=int)
    center = np.array(buffer.shape, dtype=int) // 2
    selection = create_selection_from_center(center, arr.shape)
    buffer[selection] = arr.copy()
    return buffer


def create_selection_from_center(center: ndarray2, shape: ndarray2) -> tuple[slice]:
    shape = np.array(shape, dtype=int)
    top_left = center - shape // 2
    bottom_right = top_left + shape
    return slice(top_left[0], bottom_right[0]), slice(top_left[1], bottom_right[1])


def correlate_and_combine(
    current_frame: ndarray2,
    next_frame: ndarray2,
    interrogation_window_shape: ndarray2 | tuple[int],
    search_window_shape: ndarray2 | tuple[int],
) -> tuple[ndarray4, ndarray3]:
    """computes correlations for image pair, returns the correlations and peak offsets in ndarrays.



    Args:
        current_frame (ndarray2): _description_
        next_frame (ndarray2): _description_
        interrogation_window_shape: ndarray2,
        search_window_shape: ndarray2,

    Returns:
        tuple[ndarray4, ndarray3]: The first returned item is the correlation matrices with (grid_y, grid_x, [cor_y, cor_x]).
        The second item is the peak value offsets to the patch centers, with (grid_y, grid_x, (offset_y, offset_x))
    """
    interrogation_window_shape = np.array(interrogation_window_shape, dtype=int)
    search_window_shape = np.array(search_window_shape, dtype=int)

    correlation_patch_result_size = search_window_shape - interrogation_window_shape + 1
    window_grid_resolution: ndarray2 = current_frame.shape // interrogation_window_shape
    correlation_results: ndarray4 = np.zeros(np.concatenate((window_grid_resolution, correlation_patch_result_size)))

    # Kernel center grid coords -> pixel coords  # (h, w, (pixel_y, pixel_x))
    # E.g. kernel_centers[2, 3] -> [64, 96]
    kernel_centers: ndarray3 = np.zeros([window_grid_resolution[0], window_grid_resolution[1], 2], dtype=int)
    center_ys = np.arange(interrogation_window_shape[0] // 2, current_frame.shape[0], interrogation_window_shape[0])
    center_xs = np.arange(interrogation_window_shape[1] // 2, current_frame.shape[1], interrogation_window_shape[1])
    grid_ys, grid_xs = np.meshgrid(center_ys, center_xs)
    kernel_centers = np.c_[np.expand_dims(grid_ys, -1), np.expand_dims(grid_xs, -1)]

    correlation_maximum_offsets: ndarray3 = np.zeros_like(kernel_centers, dtype=float32)  # (y, x, [dy, dx])

    for grid_coord in np.ndindex(kernel_centers.shape[:-1]):
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

    return correlation_results, correlation_maximum_offsets


def normalize(x: ndarray) -> ndarray:
    return (x - x.min()) / (x.max() - x.min())


def main():
    # Load images
    current_frames = np.load("./stkA.npy")  # (D, H, W)
    next_frames = np.load("./stkB.npy")  # (D, H, W)
    D, H, W = current_frames.shape
    print(f'loaded image with shape {(D, H, W)}')

    correlation_2d_layout_stack = []
    velocity_stack = []
    velocity_x_stack = []

    # (z, grid_y, grid_x, [y, x]) ->   (z, [y, x])
    # Iterate over possible dz values
    for depth, (current_frame, next_frame) in alive_it(enumerate(zip(current_frames, next_frames))):
        correlation_matrix, peak_offsets = correlate_and_combine(
            current_frame=current_frame,
            next_frame=next_frame,
            interrogation_window_shape=(80, 80), # 32, 64; 80, 120
            search_window_shape=(120, 120),
        )

        # print(f"correlation matrix size: {correlation_matrix.shape}")

        # Save output
        # Normalize to [0, 1]
        normalized_correlation = normalize(correlation_matrix)
        correlation_2d_layout_stack.append(correlation_matrix)
        h, w, y, x = normalized_correlation.shape

        output_2d_layout = normalized_correlation.transpose((0, 2, 1, 3)).reshape((h * y, w * x))

        # Save as a 32-bit floating point TIFF image
        os.makedirs(output_dir, exist_ok=True)
        PIL.Image.fromarray(output_2d_layout.__mul__(255).astype(np.uint8), mode="L").save(
            os.path.join(output_dir, f"dz_{depth * 10}.tif")
        )

        # Horizontal velocity profile -- The x component, so the index is 1 instead of 0.
        # v_x = peak_offsets[:, :, 1].mean()
        v_x = np.mean(peak_offsets[:, :, 1])
        # v_median = np.median(peak_offsets[:, :, 1])

        velocity_x_stack.append(v_x)
        velocity_stack.append(peak_offsets)

    correlation_stack: ndarray5 = np.stack(correlation_2d_layout_stack)
    np.save('corr_matrix.npy', correlation_stack)
    # print(f'the shape of corr_matrix is:', correlation_stack.shape)
    print(velocity_x_stack)

    plot_velocity_curve(velocity_x_stack)
    # plot_zox_view(correlation_2d_layout_stack)
    # plot_zoy_view(velocity_stack)
    # plot_surface(output_2d_layout)
    plt.show()


def plot_velocity_curve(velocity_stack: list[np.ndarray]) -> plt.Axes:
    zs = list(range(len(velocity_stack)))
    vx_s = velocity_stack

    fig = plt.figure()
    _ = fig.add_subplot(111)

    # Fit a parabola
    coeffs = np.polyfit(x=zs, y=vx_s, deg=2)
    # Create a polynomial function
    poly = np.poly1d(coeffs)
    y_fit = poly(zs)

    # Plot
    plt.scatter(x=zs, y=vx_s, label="velocity profile")
    plt.plot(zs, y_fit, color="red", label="fitted curve")


    zs = np.array(zs)
    # Define the function f(x, y)
    def f(x):
        return 8 * (1 - (np.abs(x * 10 - 150) / 150) ** 2)
    plt.plot(zs, f(zs), color='green', label='raw curve')
    plt.xlabel("z")
    plt.ylabel("velocity")
    plt.title("velocity profile")
    plt.legend()
    plt.grid(True)
    return plt.gca()


def plot_zox_view(correlation_2d_layout_stack: list[np.ndarray]) -> plt.Axes:
    arr = np.array(correlation_2d_layout_stack)  # (z, grid_y, grid_x, corr_y, corr_x)
    z, h, w, _y, x = arr.shape
    zox = arr.max(axis=-2).transpose((1, 0, 2, 3)).reshape((h * z, w * x))
    zox[::z, :] = 0
    zox[:, ::x] = 0

    plt.xlabel("x")
    plt.ylabel("z")
    plt.title("zox_view")
    plt.imshow(zox, cmap='gray')
    return plt.gca()


def plot_zoy_view(velocity_stack: list[ndarray3]) -> plt.Axes:
    arr = np.array(velocity_stack)  # (z, grid_y, grid_x, (v_y, v_x))
    zoy = arr[..., 1].mean(-1)  # (z, h)
    
    plt.xlabel("x")
    plt.ylabel("z")
    plt.title("zox_view")
    plt.imshow(zoy, cmap='gray')
    return plt.gca()


def plot_surface(correlation_matrix: ndarray3) -> plt.Axes:
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    correlation_matrix = np.array(correlation_matrix)
    selected_correlation_matrix = correlation_matrix
    print(selected_correlation_matrix.shape)
    h, w = selected_correlation_matrix.shape
    Y, X = np.meshgrid(np.arange(w), np.arange(h))

    ax.plot_surface(Y, X, selected_correlation_matrix, cmap="jet", linewidth=0.2)  # type: ignore
    plt.title("Correlation map — peak is the most probable shift")
    return ax

if __name__ == "__main__":
    main()
