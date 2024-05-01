from typing import Union

import global_core as g
import numpy as np
import SimpleITK as sitk


def is_image(obj):
    return isinstance(obj, sitk.Image)


def load_as_np_array(image: Union[sitk.Image, np.ndarray]):
    if is_image(image):
        return sitk.GetArrayFromImage(image)
    else:
        return image


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
