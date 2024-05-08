from typing import Union

import global_core as g
import numpy as np
import torch.nn as nn
from medpy.metric import asd, assd, hd, hd95
from numpy import ndarray
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.spatial import cKDTree
from str_lib import Metric
from torch import Tensor


def assert_shape(test, reference):
    assert test.shape == reference.shape, "Shape mismatch: {} and {}".format(
        test.shape, reference.shape
    )


class ConfusionMatrix:
    def __init__(self, test=None, reference=None):
        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None
        self.size = None
        self.reference_empty = None
        self.reference_full = None
        self.test_empty = None
        self.test_full = None
        self.set_reference(reference)
        self.set_test(test)

    def set_test(self, test):
        self.test = test
        self.reset()

    def set_reference(self, reference):
        self.reference = reference
        self.reset()

    def reset(self):
        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None
        self.size = None
        self.test_empty = None
        self.test_full = None
        self.reference_empty = None
        self.reference_full = None

    def compute(self):
        if self.test is None or self.reference is None:
            raise ValueError(
                "'test' and 'reference' must both be set to compute confusion matrix."
            )

        assert_shape(self.test, self.reference)

        self.tp = int(((self.test != 0) * (self.reference != 0)).sum())
        self.fp = int(((self.test != 0) * (self.reference == 0)).sum())
        self.tn = int(((self.test == 0) * (self.reference == 0)).sum())
        self.fn = int(((self.test == 0) * (self.reference != 0)).sum())
        self.size = int(np.prod(self.reference.shape, dtype=np.int64))
        self.test_empty = not np.any(self.test)
        self.test_full = np.all(self.test)
        self.reference_empty = not np.any(self.reference)
        self.reference_full = np.all(self.reference)

    def get_matrix(self):
        for entry in (self.tp, self.fp, self.tn, self.fn):
            if entry is None:
                self.compute()
                break

        return self.tp, self.fp, self.tn, self.fn

    def get_size(self):
        if self.size is None:
            self.compute()
        return self.size

    def get_existence(self):
        for case in (
            self.test_empty,
            self.test_full,
            self.reference_empty,
            self.reference_full,
        ):
            if case is None:
                self.compute()
                break

        return (
            self.test_empty,
            self.test_full,
            self.reference_empty,
            self.reference_full,
        )


def dice(
    test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs
):
    """2TP / (2TP + FP + FN)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    (
        test_empty,
        test_full,
        reference_empty,
        reference_full,
    ) = confusion_matrix.get_existence()

    if test_empty and reference_empty:
        return 1.0
    elif test_empty or test_full or reference_empty or reference_full:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.0

    return float(2.0 * tp / (2 * tp + fp + fn))


def jaccard(
    test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs
):
    """TP / (TP + FP + FN)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    (
        test_empty,
        test_full,
        reference_empty,
        reference_full,
    ) = confusion_matrix.get_existence()

    if test_empty and reference_empty:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.0

    return float(tp / (tp + fp + fn))


def precision(
    test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs
):
    """TP / (TP + FP)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    (
        test_empty,
        test_full,
        reference_empty,
        reference_full,
    ) = confusion_matrix.get_existence()

    if test_empty:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.0

    return float(tp / (tp + fp))


def sensitivity(
    test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs
):
    """TP / (TP + FN)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    (
        test_empty,
        test_full,
        reference_empty,
        reference_full,
    ) = confusion_matrix.get_existence()

    if reference_empty:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.0

    return float(tp / (tp + fn))


def recall(
    test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs
):
    """TP / (TP + FN)"""

    return sensitivity(test, reference, confusion_matrix, nan_for_nonexisting, **kwargs)


def specificity(
    test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs
):
    """TN / (TN + FP)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    (
        test_empty,
        test_full,
        reference_empty,
        reference_full,
    ) = confusion_matrix.get_existence()

    if reference_full:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.0

    return float(tn / (tn + fp))


def accuracy(test=None, reference=None, confusion_matrix=None, **kwargs):
    """(TP + TN) / (TP + FP + FN + TN)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()

    return float((tp + tn) / (tp + fp + tn + fn))


def fscore(
    test=None,
    reference=None,
    confusion_matrix=None,
    nan_for_nonexisting=True,
    beta=1.0,
    **kwargs
):
    """(1 + b^2) * TP / ((1 + b^2) * TP + b^2 * FN + FP)"""

    precision_ = precision(test, reference, confusion_matrix, nan_for_nonexisting)
    recall_ = recall(test, reference, confusion_matrix, nan_for_nonexisting)

    return (
        (1 + beta * beta)
        * precision_
        * recall_
        / ((beta * beta * precision_) + recall_)
    )


def false_positive_rate(
    test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs
):
    """FP / (FP + TN)"""

    return 1 - specificity(test, reference, confusion_matrix, nan_for_nonexisting)


def false_omission_rate(
    test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs
):
    """FN / (TN + FN)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    (
        test_empty,
        test_full,
        reference_empty,
        reference_full,
    ) = confusion_matrix.get_existence()

    if test_full:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.0

    return float(fn / (fn + tn))


def false_negative_rate(
    test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs
):
    """FN / (TP + FN)"""

    return 1 - sensitivity(test, reference, confusion_matrix, nan_for_nonexisting)


def true_negative_rate(
    test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs
):
    """TN / (TN + FP)"""

    return specificity(test, reference, confusion_matrix, nan_for_nonexisting)


def false_discovery_rate(
    test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs
):
    """FP / (TP + FP)"""

    return 1 - precision(test, reference, confusion_matrix, nan_for_nonexisting)


def negative_predictive_value(
    test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs
):
    """TN / (TN + FN)"""

    return 1 - false_omission_rate(
        test, reference, confusion_matrix, nan_for_nonexisting
    )


def total_positives_test(test=None, reference=None, confusion_matrix=None, **kwargs):
    """TP + FP"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()

    return tp + fp


def total_negatives_test(test=None, reference=None, confusion_matrix=None, **kwargs):
    """TN + FN"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()

    return tn + fn


def total_positives_reference(
    test=None, reference=None, confusion_matrix=None, **kwargs
):
    """TP + FN"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()

    return tp + fn


def total_negatives_reference(
    test=None, reference=None, confusion_matrix=None, **kwargs
):
    """TN + FP"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()

    return tn + fp


def hausdorff_distance(
    test=None,
    reference=None,
    confusion_matrix=None,
    nan_for_nonexisting=True,
    voxel_spacing=None,
    connectivity=1,
    **kwargs
):
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    (
        test_empty,
        test_full,
        reference_empty,
        reference_full,
    ) = confusion_matrix.get_existence()

    if test_empty or test_full or reference_empty or reference_full:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0

    test, reference = confusion_matrix.test, confusion_matrix.reference

    return hd(test, reference, voxel_spacing, connectivity)


def hausdorff_distance_95(
    test=None,
    reference=None,
    confusion_matrix=None,
    none_for_nonexisting=True,
    voxel_spacing=None,
    connectivity=1,
    **kwargs
):
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    (
        test_empty,
        test_full,
        reference_empty,
        reference_full,
    ) = confusion_matrix.get_existence()

    if test_empty and reference_empty:
        return 0.0
    elif test_empty or test_full or reference_empty or reference_full:
        if none_for_nonexisting:
            return None
        else:
            return 0.0

    test, reference = confusion_matrix.test, confusion_matrix.reference

    return hd95(test, reference, voxel_spacing, connectivity)


def avg_surface_distance(
    test=None,
    reference=None,
    confusion_matrix=None,
    nan_for_nonexisting=True,
    voxel_spacing=None,
    connectivity=1,
    **kwargs
):
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    (
        test_empty,
        test_full,
        reference_empty,
        reference_full,
    ) = confusion_matrix.get_existence()

    if test_empty or test_full or reference_empty or reference_full:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0

    test, reference = confusion_matrix.test, confusion_matrix.reference

    return asd(test, reference, voxel_spacing, connectivity)


def avg_surface_distance_symmetric(
    test=None,
    reference=None,
    confusion_matrix=None,
    none_for_nonexisting=True,
    voxel_spacing=None,
    connectivity=1,
    **kwargs
):
    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    (
        test_empty,
        test_full,
        reference_empty,
        reference_full,
    ) = confusion_matrix.get_existence()

    if test_empty and reference_empty:
        return 0.0
    elif test_empty or test_full or reference_empty or reference_full:
        if none_for_nonexisting:
            return None
        else:
            return 0.0

    test, reference = confusion_matrix.test, confusion_matrix.reference

    return assd(test, reference, voxel_spacing, connectivity)


ALL_METRICS = {
    "False Positive Rate": false_positive_rate,
    "Dice": dice,
    "Jaccard": jaccard,
    "Hausdorff Distance": hausdorff_distance,
    "Hausdorff Distance 95": hausdorff_distance_95,
    "Precision": precision,
    "Recall": recall,
    "Avg. Symmetric Surface Distance": avg_surface_distance_symmetric,
    "Avg. Surface Distance": avg_surface_distance,
    "Accuracy": accuracy,
    "False Omission Rate": false_omission_rate,
    "Negative Predictive Value": negative_predictive_value,
    "False Negative Rate": false_negative_rate,
    "True Negative Rate": true_negative_rate,
    "False Discovery Rate": false_discovery_rate,
    "Total Positives Test": total_positives_test,
    "Total Negatives Test": total_negatives_test,
    "Total Positives Reference": total_positives_reference,
    "total Negatives Reference": total_negatives_reference,
}


# Define the function find_surface_pixels
def find_surface_pixels(binary_img):
    """Find surface (boundary) pixels in the binary image using morphological operations."""

    binary_img = binary_img.astype(bool)
    dilated_img = binary_dilation(binary_img)
    eroded_img = binary_erosion(binary_img)
    surface_img = dilated_img != eroded_img
    return np.argwhere(surface_img)


def surface_distances(binary_img_1, binary_img_2, spacing=(1, 1, 1)):
    """
    Calculate distances from each surface pixel in binary_img_1 to the nearest surface point in binary_img_2.

    :param binary_img_1: First binary image (numpy array).
    :param binary_img_2: Second binary image (numpy array).
    :param spacing: Tuple representing the pixel spacing in each dimension (dz, dy, dx).
    :return: Array of distances.
    """
    # Get the surface pixels for both images
    # surface_points_1 = g.find_contours(binary_img_1)
    # surface_points_2 = g.find_contours(binary_img_2)
    surface_points_1 = find_surface_pixels(binary_img_1)
    surface_points_2 = find_surface_pixels(binary_img_2)

    # Adjust points by the spacing for accurate distance calculation
    surface_points_1 = surface_points_1 * np.array(spacing)
    surface_points_2 = surface_points_2 * np.array(spacing)

    # Use KDTree for efficient nearest neighbor calculation
    tree = cKDTree(surface_points_2)
    distances, _ = tree.query(surface_points_1)

    return distances


# only for inference
class SegmentationMetric(nn.Module):
    def __init__(
        self,
        metric: str,  # dsc/msd/hd95
    ):
        super().__init__()
        self.__metric = metric
        if self.__metric not in [Metric.DSC, Metric.MSD, Metric.HD95]:
            g.error_exit("Value of metric must be one of dsc/msd/hd95!")

    def forward(self, preds: Union[Tensor, ndarray], labels: Union[Tensor, ndarray]):
        if len(preds.shape) == 4:
            preds = preds.squeeze()
        else:
            pass

        if len(labels.shape) == 4:
            labels = labels.squeeze()
        else:
            pass

        preds = g.binarize_img(preds)
        labels = g.binarize_img(labels)

        if isinstance(preds, Tensor):
            preds = preds.cpu().numpy()
        else:
            pass

        if isinstance(labels, Tensor):
            labels = labels.cpu().numpy()
        else:
            pass

        if self.__metric == Metric.DSC:
            return dice(
                test=preds,
                reference=labels,
                nan_for_nonexisting=False,
            )

        elif self.__metric == Metric.MSD:
            return avg_surface_distance_symmetric(
                test=preds,
                reference=labels,
                none_for_nonexisting=True,
                voxel_spacing=g.NII_SPACING,
            )

        elif self.__metric == Metric.HD95:
            return hausdorff_distance_95(
                test=preds,
                reference=labels,
                none_for_nonexisting=True,
                voxel_spacing=g.NII_SPACING,
            )
