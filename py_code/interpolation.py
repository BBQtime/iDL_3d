import numpy as np
from scipy.ndimage import (
    binary_erosion,
    distance_transform_edt,
    generate_binary_structure,
)


def __interpolate_with_erosion(non_empty_slice, steps, erode=True):
    interpolated_slices = np.zeros(
        (steps, *non_empty_slice.shape), dtype=non_empty_slice.dtype
    )
    structure = generate_binary_structure(2, 1)  # 2D connectivity structure

    # Adjust how the erosion steps are calculated to ensure the shape persists longer
    # Consider the size of the non_empty_slice to adjust erosion rate dynamically
    total_area = np.sum(non_empty_slice)
    # Calculate a factor that reduces erosion speed based on the initial shape size and desired steps
    erosion_factor = np.clip((total_area / (steps**2)), 0.1, 1)

    # Initialize a variable to track the adjusted erosion progress
    erosion_progress = 0

    for step in range(steps):
        if erode:
            erosion_progress += erosion_factor
        else:
            erosion_progress = max(0, erosion_progress - erosion_factor)

        # Apply the adjusted erosion based on the cumulative progress
        iterations = int(np.round(erosion_progress))

        if iterations > 0:
            eroded_slice = binary_erosion(
                non_empty_slice, structure, iterations=iterations
            )
        else:
            eroded_slice = non_empty_slice

        interpolated_slices[step] = eroded_slice.astype(non_empty_slice.dtype)

    return interpolated_slices


def interpolate_shapes(start_slice, end_slice, steps):
    # Both slices are empty
    if not np.any(start_slice) and not np.any(end_slice):
        return np.zeros((steps, *start_slice.shape), dtype=start_slice.dtype)

    # Only start_slice is empty, erode end_slice to disappear
    if not np.any(start_slice) and np.any(end_slice):
        return __interpolate_with_erosion(end_slice, steps, erode=True)[::-1]

    # Only end_slice is empty, reverse erosion (start from a single point and expand)
    if np.any(start_slice) and not np.any(end_slice):
        return __interpolate_with_erosion(start_slice, steps, erode=True)

    # Normal case: both slices are non-empty, interpolate between them
    sdt_start = distance_transform_edt(start_slice) - distance_transform_edt(
        1 - start_slice
    )
    sdt_end = distance_transform_edt(end_slice) - distance_transform_edt(1 - end_slice)

    interpolated_slices = np.zeros((steps, *start_slice.shape), dtype=start_slice.dtype)
    for step in range(steps):
        fraction = (step + 1) / (steps + 1)
        sdt_interpolated = (1 - fraction) * sdt_start + fraction * sdt_end
        interpolated_slice = (sdt_interpolated > 0).astype(start_slice.dtype)
        interpolated_slices[step] = interpolated_slice

    return interpolated_slices
