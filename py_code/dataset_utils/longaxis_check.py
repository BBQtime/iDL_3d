import os

import nibabel as nib
import numpy as np


def check_gravity_centers_within_masks(gtvn_folder):
    """
    Check if gravity centers are within the GTVn masks.

    Parameters:
    gtvn_folder (str): Path to the folder containing the GTVn maps.
    gravity_center_folder (str): Path to the folder containing the gravity center maps.
    """
    # Loop through all files in the GTVn folder
    for file_name in os.listdir(gtvn_folder):
        if file_name.endswith("_GTVn.nii.gz"):
            gtvn_path = os.path.join(gtvn_folder, file_name)

            gravity_center_path = gtvn_path.replace(".nii.gz", "_clicks.nii.gz")

            if not os.path.exists(gravity_center_path):
                print(f"Gravity center map missing for: {file_name}")
                continue

            # Load the GTVn mask and gravity center map
            gtvn_img = nib.load(gtvn_path)
            gravity_center_img = nib.load(gravity_center_path)

            gtvn_data = gtvn_img.get_fdata().astype(np.uint8)
            gravity_center_data = gravity_center_img.get_fdata().astype(np.uint8)

            # Find the coordinates of the gravity centers
            gravity_centers = np.argwhere(gravity_center_data == 1)

            # Check if each gravity center is within the mask
            for center in gravity_centers:
                if gtvn_data[tuple(center)] == 0:
                    print(f"Gravity center outside mask for: {file_name}, at {center}")


# Example usage:
gtvn_folder = r"E:\Jasper\HECKTOR_2022\data"  # Replace with the actual path


check_gravity_centers_within_masks(gtvn_folder)
