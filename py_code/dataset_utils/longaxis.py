import fnmatch
import os

import cc3d
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import center_of_mass
from scipy.spatial import distance
from sklearn.decomposition import PCA


def binarize_img(img, threshold=0.5):

    img = np.asarray(img)
    # Apply threshold to binarize the image
    binary_img = (img > threshold).astype(np.uint8)

    return binary_img


def pca_long_axis(cur_cc):
    """Perform PCA on the connected component to find the long axis."""
    coords = np.argwhere(cur_cc)  # Get coordinates of the component
    pca = PCA(n_components=3)  # 3D PCA
    pca.fit(coords)

    # The first principal component is the direction of the long axis
    long_axis_vector = pca.components_[0]
    return long_axis_vector, coords


def compute_long_axis_diameter_pca(coords, long_axis_vector):
    """Compute the length along the long axis by projecting onto the axis."""
    projected_coords = np.dot(coords, long_axis_vector)
    return projected_coords.max() - projected_coords.min()


def split_component_pca(cur_cc, coords, long_axis_vector):
    """Split the component along the longest axis found via PCA."""
    projected_coords = np.dot(coords, long_axis_vector)
    median_value = np.median(projected_coords)

    half1 = np.zeros_like(cur_cc)
    half2 = np.zeros_like(cur_cc)

    for coord, proj in zip(coords, projected_coords):
        if proj <= median_value:
            half1[tuple(coord)] = cur_cc[tuple(coord)]
        else:
            half2[tuple(coord)] = cur_cc[tuple(coord)]

    return half1, half2


def find_nearest_component_voxel(center, coords):
    """Find the nearest voxel in the component to the given center of gravity."""
    distances = distance.cdist([center], coords)
    nearest_index = distances.argmin()
    nearest_voxel = coords[nearest_index]
    return nearest_voxel


def process_and_place_centers(cur_cc, threshold, gravity_centers):
    """Recursive function to process a connected component and place gravity centers."""
    long_axis_vector, coords = pca_long_axis(cur_cc)
    long_axis_diameter = compute_long_axis_diameter_pca(coords, long_axis_vector)

    if long_axis_diameter > threshold:
        # Split and recursively process the halves
        half1, half2 = split_component_pca(cur_cc, coords, long_axis_vector)
        if np.any(half1):
            process_and_place_centers(half1, threshold, gravity_centers)
        if np.any(half2):
            process_and_place_centers(half2, threshold, gravity_centers)
    else:
        # Calculate the gravity center
        gravity_center = np.array(center_of_mass(cur_cc))
        gravity_center_rounded = tuple(np.round(gravity_center).astype(int))

        # Check if the gravity center is within the component
        if cur_cc[gravity_center_rounded] != 1:
            # If not, find the nearest voxel within the component
            gravity_center_rounded = tuple(
                find_nearest_component_voxel(gravity_center, coords)
            )

        # Place a 1 at the correct location in the gravity_centers array
        gravity_centers[gravity_center_rounded] = 1


def process_connected_components_pca(img, diameter_threshold):
    img = binarize_img(img)
    all_cc, num_cc = cc3d.connected_components(img, connectivity=18, return_N=True)
    print(f"number of ccs> {(num_cc)}")
    gravity_centers = np.zeros_like(img)

    for segid in range(1, num_cc + 1):
        cur_cc = (all_cc == segid).astype(np.uint8)
        process_and_place_centers(cur_cc, diameter_threshold, gravity_centers)

    return gravity_centers


folder = r"E:\Jasper\HECKTOR_2022\data"
matching_files = []
pattern = "*_GTVn.nii.gz"
# Example usage:
diameter_threshold = 30  # Adjust this threshold based on your requirements
outdir = r"E:\Jasper\HECKTOR_2022\data"

# Walk through directory
for dirpath, _, filenames in os.walk(folder):
    for filename in fnmatch.filter(filenames, pattern):
        matching_files.append(os.path.join(dirpath, filename))

for file in matching_files:
    print(file)
    img = sitk.ReadImage(file)
    spacing = img.GetSpacing()
    origin = img.GetOrigin()
    imgarray = sitk.GetArrayFromImage(img)
    gravity_centers = process_connected_components_pca(imgarray, diameter_threshold)
    print(f"number of gcs> {np.sum(gravity_centers)}")
    itk_img = sitk.GetImageFromArray(gravity_centers)
    itk_img.SetSpacing(spacing)
    itk_img.SetOrigin(origin)
    filename = os.path.basename(file)

    # outfile = os.path.join(outdir, filename)
    # sitk.WriteImage(img, outfile)
    filename = filename.replace(".nii.gz", "_clicks.nii.gz")
    outfile = os.path.join(outdir, filename)
    sitk.WriteImage(itk_img, outfile)
