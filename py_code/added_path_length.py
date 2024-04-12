from typing import Union

import numpy as np
import SimpleITK as sitk
from custom import Global as g

# from simplestruct.filters import generate_edge_of_structure
# from simplestruct.utils.type_functions import is_image, load_as_np_array


def is_image(obj):
    return isinstance(obj, sitk.Image)


def load_as_np_array(image: Union[sitk.Image, np.ndarray]):
    if is_image(image):
        return sitk.GetArrayFromImage(image)
    else:
        return image


# def generate_edge_of_structure(
#     structure: np.ndarray, use_2d: bool = True
# ) -> np.ndarray:
#     """
#     Is binarized, so 0 is background and 1-n is contour. Array is ordered [z, y, x]
#     :param mask:
#     :return:
#     """

#     mask = structure != 0
#     edge = np.zeros_like(mask)
#     for z in range(0, mask.shape[0]):
#         for y in range(0, mask.shape[1]):
#             for x in range(0, mask.shape[2]):
#                 if use_2d:
#                     i_sum = np.sum(mask[z, y - 1 : y + 2, x - 1 : x + 2])
#                     if i_sum < 9:
#                         edge[z, y, x] = mask[z, y, x]
#                 else:
#                     i_sum = np.sum(mask[z - 1 : z + 2, y - 1 : y + 2, x - 1 : x + 2])
#                     if i_sum < 27:
#                         edge[z, y, x] = mask[z, y, x]

#     return edge


class APL:
    def __init__(
        self,
        reference_structure: Union[sitk.Image, np.ndarray],
        other_structure: Union[sitk.Image, np.ndarray],
    ):
        """
        It must be possible to cast contour images as boolean arrays - only 0 and one other label should be present.
        """
        self.reference_structure = load_as_np_array(reference_structure)
        self.other_structure = load_as_np_array(other_structure)

        self.gt_edge = None
        self.other_edge = None

    def execute(self):
        # self.gt_edge = generate_edge_of_structure(self.reference_structure)
        # self.other_edge = generate_edge_of_structure(self.other_structure)
        self.gt_edge = g.find_contours(self.reference_structure)
        self.other_edge = g.find_contours(self.other_structure)

        # Nii.save(self.reference_structure, os.path.join(g.DEBUG_DIR, "origin.nii.gz"))
        # Nii.save(self.gt_edge, os.path.join(g.DEBUG_DIR, "contour.nii.gz"))

        ## Edge case if prediction is all false and should not be. If so, return full size of prediction
        if np.count_nonzero(self.gt_edge) == 0:
            self.raw_apl = np.count_nonzero(self.other_edge)
        else:
            self.raw_apl = (self.gt_edge < self.other_edge).astype(int).sum()

        self.norm_apl = self.raw_apl / np.count_nonzero(self.other_edge)

    def get_apl(self, normalized=True):
        # make sure execute() only run once
        if self.gt_edge is None:
            self.execute()

        if normalized:
            return self.norm_apl
        else:
            return self.raw_apl
