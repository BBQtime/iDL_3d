import global_core as g
import numpy as np


def find_distance_for_coord(
    coord: np.ndarray, other_coords: np.ndarray, spacing_array: np.ndarray
):
    vectors = np.zeros(other_coords.shape, dtype=float)
    vectors[:, 0] = other_coords[:, 0] - coord[0]
    vectors[:, 1] = other_coords[:, 1] - coord[1]
    vectors[:, 2] = other_coords[:, 2] - coord[2]

    vectors[:, 0] = vectors[:, 0] * spacing_array[0]
    vectors[:, 1] = vectors[:, 1] * spacing_array[1]
    vectors[:, 2] = vectors[:, 2] * spacing_array[2]

    # Euclidian length
    vectors = np.power(vectors, 2)
    vectors = np.sum(vectors, axis=1)
    vector_lengths = np.sqrt(vectors)

    coord_hd = np.min(vector_lengths)

    return coord_hd


class HD:
    def __init__(self, reference_image: np.ndarray, other_image: np.ndarray):
        """
        Contours will be cast to bool
        """
        self.reference_image = reference_image
        self.other_image = other_image

        # self.ref_arr = generate_edge_of_structure(
        #     sitk.GetArrayFromImage(self.reference_image), use_2d=False
        # )
        # self.other_arr = generate_edge_of_structure(
        #     sitk.GetArrayFromImage(self.other_image), use_2d=False
        # )
        self.ref_arr = g.find_contours(self.reference_image)
        self.other_arr = g.find_contours(self.other_image)

        # self.spacing_arr = np.array(self.reference_image.GetSpacing())[-1::-1]
        self.spacing_arr = np.array(g.NII_SPACING)[-1::-1]

        self.distance_matrix_ref_to_other = None
        self.distance_matrix_other_to_ref = None

    def _calculate_distance_matrix(
        self, reference: np.ndarray, other: np.ndarray
    ) -> np.ndarray:
        """
        This function gives the directed distance from reference to other contour.
        :return:
        """
        ref_coords = np.argwhere(reference)
        other_coords = np.argwhere(other)

        distance_matrix = np.empty(
            (ref_coords.shape[0], 4)
        )  # columns are Z, Y, X, hausdorff distance for this point
        distance_matrix[:, :3] = ref_coords
        for i in range(0, ref_coords.shape[0]):
            distance_matrix[i, 3] = find_distance_for_coord(
                coord=ref_coords[i, :],
                other_coords=other_coords,
                spacing_array=self.spacing_arr,
            )

        return distance_matrix

    def execute(self, undirected=True):
        if self.distance_matrix_ref_to_other is None:
            self.distance_matrix_ref_to_other = self._calculate_distance_matrix(
                self.ref_arr, self.other_arr
            )
        if undirected and self.distance_matrix_other_to_ref is None:
            self.distance_matrix_other_to_ref = self._calculate_distance_matrix(
                self.other_arr, self.ref_arr
            )

    def get_distances(self, undirected=True):
        self.execute(undirected=undirected)
        if undirected:
            return np.concatenate(
                [
                    self.distance_matrix_ref_to_other[:, 3],
                    self.distance_matrix_other_to_ref[:, 3],
                ]
            )
        else:
            return self.distance_matrix_ref_to_other[:, 3]

    def get_distance_matrix_ref_to_other(self):
        self.execute(undirected=False)
        return self.distance_matrix_ref_to_other

    def func_on_min_distances(self, func, undirected=True):
        return func(self.get_distances(undirected))

    def get_max_min_hd(self, undirected=True):
        return self.func_on_min_distances(func=np.max, undirected=undirected)

    def get_avg_min_hd(self, undirected=True):
        return self.func_on_min_distances(func=np.mean, undirected=undirected)

    def get_percentile_min_hd(self, percentile=0.95, undirected=True):
        arr = self.func_on_min_distances(np.sort, undirected=undirected)
        return arr[int(arr.shape[0] * percentile)]


class SurfaceDice:
    def __init__(
        self,
        reference_image: np.ndarray,
        other_image: np.ndarray,
    ):
        self.hd = HD(reference_image=reference_image, other_image=other_image)
        self.distances = None

    def execute(self):
        self.distances = self.hd.get_distances(undirected=False)

    def get_surface_dice(self, tolerance: float = 1):
        if self.distances is None:
            self.execute()
        under_tolerance = self.distances <= tolerance
        surface_dice = np.count_nonzero(under_tolerance) / self.distances.shape[0]
        return surface_dice
