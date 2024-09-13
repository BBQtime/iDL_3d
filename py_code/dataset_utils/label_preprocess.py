import fnmatch
import os
import shutil
from pathlib import Path

import cc3d
import global_utils.global_core as g
import numpy as np
import SimpleITK as sitk
from global_utils.str_lib import DatasetVer
from scipy.ndimage import center_of_mass
from scipy.spatial import distance
from sklearn.decomposition import PCA
from tqdm import tqdm


def __binarize_img(img, threshold=0.5):

    img = np.asarray(img)
    # Apply threshold to binarize the image
    binary_img = (img > threshold).astype(np.uint8)

    return binary_img


def __pca_long_axis(cur_cc):
    """Perform PCA on the connected component to find the long axis."""
    coords = np.argwhere(cur_cc)  # Get coordinates of the component

    if coords.shape[0] < 3:
        # If there are fewer than 3 points, adjust the number of components to be the number of points available
        pca = PCA(n_components=min(coords.shape[0], 3))
    else:
        pca = PCA(n_components=3)  # 3D PCA if there are 3 or more points

    pca.fit(coords)

    # The first principal component is the direction of the long axis
    long_axis_vector = pca.components_[0]
    return long_axis_vector, coords


def __compute_long_axis_diameter_pca(coords, long_axis_vector):
    """Compute the length along the long axis by projecting onto the axis."""
    projected_coords = np.dot(coords, long_axis_vector)
    return projected_coords.max() - projected_coords.min()


def __split_component_pca(cur_cc, coords, long_axis_vector):
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


def __find_nearest_component_voxel(center, coords):
    """Find the nearest voxel in the component to the given center of gravity."""
    distances = distance.cdist([center], coords)
    nearest_index = distances.argmin()
    nearest_voxel = coords[nearest_index]
    return nearest_voxel


def __process_and_place_centers(cur_cc, threshold, gravity_centers):
    """Recursive function to process a connected component and place gravity centers."""
    long_axis_vector, coords = __pca_long_axis(cur_cc)
    long_axis_diameter = __compute_long_axis_diameter_pca(coords, long_axis_vector)

    if long_axis_diameter > threshold:
        # Split and recursively process the halves
        half1, half2 = __split_component_pca(cur_cc, coords, long_axis_vector)
        if np.any(half1):
            __process_and_place_centers(half1, threshold, gravity_centers)
        if np.any(half2):
            __process_and_place_centers(half2, threshold, gravity_centers)
    else:
        # Calculate the gravity center
        gravity_center = np.array(center_of_mass(cur_cc))
        gravity_center_rounded = tuple(np.round(gravity_center).astype(int))

        # Check if the gravity center is within the component
        if cur_cc[gravity_center_rounded] != 1:
            # If not, find the nearest voxel within the component
            gravity_center_rounded = tuple(
                __find_nearest_component_voxel(gravity_center, coords)
            )

        # Place a 1 at the correct location in the gravity_centers array
        gravity_centers[gravity_center_rounded] = 1


def __process_connected_components_pca(img, diameter_threshold):
    img = __binarize_img(img)
    all_cc, num_cc = cc3d.connected_components(img, connectivity=18, return_N=True)
    print(f"number of ccs> {(num_cc)}")
    gravity_centers = np.zeros_like(img)

    for segid in range(1, num_cc + 1):
        cur_cc = (all_cc == segid).astype(np.uint8)
        __process_and_place_centers(cur_cc, diameter_threshold, gravity_centers)

    return gravity_centers


def generate_gtvn_clicks_nii(dataset_ver: str):
    diameter_threshold = 30  # Adjust this threshold based on your requirements

    dataset_dir = g.DATASET_DIR[dataset_ver]
    if dataset_ver in [DatasetVer.AU, DatasetVer.OBS_STUDY]:
        gtvn_path_list = g.get_sub_files(
            dataset_dir, key_word="_GTVn.nii", full_path=True
        )
    elif dataset_ver in [DatasetVer.AU_EXT, DatasetVer.MDA]:
        gtvn_path_list = []
        patient_dir_list = g.get_sub_dirs(dataset_dir, full_path=True)
        for patient_dir in patient_dir_list:
            gtvn_path = g.get_sub_files(
                patient_dir, key_word="GTVn.nii", full_path=True
            )
            if len(gtvn_path) > 0:
                gtvn_path_list.append(gtvn_path[0])
            else:
                print(Path(patient_dir).name, " has no GTVn")
    else:
        g.error_exit("dataset error")

    for gtvn_path in tqdm(gtvn_path_list):
        print(gtvn_path)
        img = sitk.ReadImage(gtvn_path)
        spacing = img.GetSpacing()
        origin = img.GetOrigin()
        imgarray = sitk.GetArrayFromImage(img)
        gravity_centers = __process_connected_components_pca(
            imgarray, diameter_threshold
        )
        print(f"number of gcs> {np.sum(gravity_centers)}")
        itk_img = sitk.GetImageFromArray(gravity_centers)
        itk_img.SetSpacing(spacing)
        itk_img.SetOrigin(origin)

        # save clicks
        gtvn_nii_name = Path(gtvn_path).name
        if gtvn_nii_name.endswith("GTVn.nii"):
            gtvn_nii_name = gtvn_nii_name.replace("GTVn.nii", "GTVn_clicks.nii.gz")
        elif gtvn_nii_name.endswith("GTVn.nii.gz"):
            gtvn_nii_name = gtvn_nii_name.replace("GTVn.nii.gz", "GTVn_clicks.nii.gz")
        gtvn_click_path = os.path.join(Path(gtvn_path).parent, gtvn_nii_name)
        sitk.WriteImage(itk_img, gtvn_click_path)


def check_gtvn_clicks_within_label(dataset_ver: str):
    dataset_dir = g.DATASET_DIR[dataset_ver]

    if dataset_ver in [DatasetVer.AU, DatasetVer.OBS_STUDY]:
        gtvn_path_list = g.get_sub_files(
            dataset_dir, key_word="GTVn.nii", full_path=True
        )
    elif dataset_ver in [DatasetVer.AU_EXT, DatasetVer.MDA]:
        gtvn_path_list = []
        patient_dir_list = g.get_sub_dirs(dataset_dir, full_path=True)
        for patient_dir in patient_dir_list:
            gtvn_path = g.get_sub_files(
                patient_dir, key_word="GTVn.nii", full_path=True
            )
            if len(gtvn_path) > 0:
                gtvn_path_list.append(gtvn_path[0])
            # else:
            #     print(Path(patient_dir).name, " has no GTVn")

    # Loop through all files in the GTVn folder
    for gtvn_path in tqdm(gtvn_path_list):
        gtvn_clicks_path = gtvn_path.replace("GTVn.nii", "GTVn_clicks.nii.gz")

        if not os.path.exists(gtvn_clicks_path):
            print("Gravity center map missing for: {}".format(gtvn_path))
            continue

        # Load the GTVn mask and gravity center map
        gtvn_img = g.load_nii(gtvn_path, binary=True).astype(np.uint8)
        gtvn_clicks_img = g.load_nii(gtvn_clicks_path, binary=True).astype(np.uint8)

        # Find the coordinates of the gravity centers
        gtvn_click_pos = np.argwhere(gtvn_clicks_img == 1)

        # Check if each gravity center is within the mask
        for cur_pos in gtvn_click_pos:
            if gtvn_img[tuple(cur_pos)] == 0:
                print(
                    "Gravity center outside mask for: {}, at {}".format(
                        gtvn_path, cur_pos
                    )
                )


def remove_label_fregments(dataset_ver: str, fregment_threshold: int = 2):

    dataset_dir = g.DATASET_DIR[dataset_ver]

    # loop through gtvt and gtvn
    for i in [
        "n",
        # "t",
    ]:
        if dataset_ver in [DatasetVer.AU, DatasetVer.OBS_STUDY]:
            label_path_list = g.get_sub_files(
                dataset_dir, key_word="GTV{}.nii".format(i), full_path=True
            )
        elif dataset_ver in [DatasetVer.AU_EXT, DatasetVer.MDA]:
            label_path_list = []
            patient_dir_list = g.get_sub_dirs(dataset_dir, full_path=True)
            for patient_dir in patient_dir_list:
                label_path = g.get_sub_files(
                    patient_dir, key_word="GTV{}.nii".format(i), full_path=True
                )
                if len(label_path) > 0:
                    label_path_list.append(label_path[0])

        for label_path in tqdm(label_path_list):
            output_dir = os.path.join(
                g.DEBUG_DIR,
                "{}_{}".format(dataset_ver, fregment_threshold),
            )
            g.create_dir(output_dir)

            # Load the NIfTI file using SimpleITK
            nii_image = sitk.ReadImage(label_path)

            # Convert the image to a numpy array for processing
            nii_array = sitk.GetArrayFromImage(nii_image)

            # Use connected component labeling to label fragments
            connected_components = sitk.ConnectedComponent(
                sitk.Cast(nii_image, sitk.sitkUInt8)
            )

            # Convert connected components image to a numpy array
            connected_array = sitk.GetArrayFromImage(connected_components)

            # Get the number of connected components
            labels = np.unique(connected_array)

            fragment_count = 0
            original_saved = False

            # Create an empty array for storing all fragments in one image
            combined_fragment_array = np.zeros_like(nii_array)

            # Process each connected component (fragment)
            for label in labels:
                if label == 0:
                    continue  # Skip the background

                # Create a mask for the current label (fragment)
                fragment_mask = connected_array == label

                # Check if the fragment has fewer or equal voxels than the threshold
                if np.sum(fragment_mask) <= fregment_threshold:
                    fragment_count += 1
                    print(f"fregment: {fragment_count}")

                    # Add this fragment to the combined fragment array
                    combined_fragment_array[fragment_mask] = 1

                    # Save the original image only once when a fragment is detected
                    if not original_saved:
                        label_name = os.path.basename(label_path).replace(".nii", "")

                        lower_threshold_file_already_exist = False
                        for lower_threshold in range(1, fregment_threshold):
                            file_to_check = os.path.join(
                                g.DEBUG_DIR,
                                "{}_{}".format(dataset_ver, lower_threshold),
                                label_name + "_origin.nii",
                            )
                            if os.path.exists(file_to_check):
                                lower_threshold_file_already_exist = True
                                break

                        if not lower_threshold_file_already_exist:
                            sitk.WriteImage(
                                nii_image,
                                os.path.join(output_dir, label_name + "_origin.nii"),
                            )

                            # also save ct img
                            ct_path = label_path.replace("GTVn.nii", "CT.nii")
                            cn_name = os.path.basename(ct_path)
                            shutil.copy(
                                ct_path,
                                os.path.join(output_dir, cn_name),
                            )
                        else:
                            print("{} already exist".format(file_to_check))

                        original_saved = True

            # If there are any fragments, save the combined fragment image
            if fragment_count > 0 and not lower_threshold_file_already_exist:
                # Convert the numpy array back to a SimpleITK image
                combined_fragment_image = sitk.GetImageFromArray(
                    combined_fragment_array
                )

                # Copy the original NIfTI image metadata (spacing, origin, direction)
                combined_fragment_image.CopyInformation(nii_image)

                # Save the combined fragment image with the fragment count in the label_name
                fragment_path = os.path.join(
                    output_dir, f"{label_name}_fragments_{fragment_count}.nii"
                )
                sitk.WriteImage(combined_fragment_image, fragment_path)
                print(f"Combined fragment file saved at: {fragment_path}")
