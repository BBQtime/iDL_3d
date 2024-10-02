import random

from global_utils.custom_dict import Dict
from monai.transforms import (
    Compose,
    Rand3DElastic,
    RandAffine,
    RandFlip,
    RandGaussianNoise,
    RandRotate,
    RandScaleIntensity,
    RandZoom,
)
from numpy import ndarray


class TraceableTransform:
    def __init__(self, transform, name=None):
        self.transform = transform
        self.name = name or transform.__class__.__name__
        self.trace = None

    def __call__(self, img):
        result = self.transform(img)
        self.trace = self._get_trace_info()  # Capture detailed trace information
        return result

    def set_random_state(self, seed):
        # If the transform has the set_random_state method, use it
        if hasattr(self.transform, "set_random_state"):
            self.transform.set_random_state(seed)

    def _get_trace_info(self):
        # Capture specific information about the transformation applied
        if isinstance(self.transform, RandRotate):

            return f"{self.name} with angles {self.transform.x}, {self.transform.y}, {self.transform.z}"
        elif isinstance(self.transform, Rand3DElastic):
            # applied = getattr(self.transform, "applied_items", None)
            sigma = getattr(self.transform, "sigma", None)
            magnitude = getattr(self.transform, "magnitude", None)
            return f"{self.name} with sigma {sigma}, magnitude {magnitude}"
        elif isinstance(self.transform, RandFlip):
            axis = getattr(self.transform.flipper, "spatial_axis", None)
            return f"{self.name} with axis {axis}"
        return f"{self.name} applied"


class RandSomeOf:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, num_transforms, seed=None):
        if seed is not None:
            # Seed the random number generator for consistent sampling
            random.seed(seed)

        # Ensure num_transforms does not exceed the number of available transforms
        num_transforms = max(1, min(num_transforms, len(self.transforms)))
        # Shuffle the transforms list to avoid any bias
        # random.shuffle(self.transforms)

        # Sample the transforms to apply
        selected_transforms = random.sample(self.transforms, num_transforms)

        # Apply seed to each selected transform
        for i, transform in enumerate(selected_transforms):
            transform.set_random_state(seed + i)

        # Compose and apply the selected transforms to the image
        composed_transform = Compose(selected_transforms)
        return composed_transform(img)


class DataAugmentation:
    def __init__(self, param: Dict = None):
        self.__transform = None
        self.rand_some_of = None  # Initialize to avoid AttributeError

        if param is None:
            self.set_no_op_transform()  # Set a no-op transform
            return

        if param["augment.pct"] <= 0 or param["augment.methods"] == []:
            self.set_no_op_transform()  # Set a no-op transform
            return

        transforms_list = []

        if "elastic" in param["augment.methods"]:
            transforms_list.append(
                TraceableTransform(
                    Rand3DElastic(
                        prob=1.0,
                        sigma_range=(8, 12),
                        magnitude_range=(50, 100),
                        spatial_size=None,
                        mode="bilinear",
                        padding_mode="zeros",
                    )
                )
            )
        if "scale" in param["augment.methods"]:
            transforms_list.append(
                TraceableTransform(
                    RandAffine(
                        prob=1.0,
                        translate_range=(0.1, 0.1, 0.1),
                        rotate_range=(0.1, 0.1, 0.1),
                        scale_range=(0.1, 0.1, 0.1),
                        padding_mode="zeros",
                    )
                )
            )

        if "translate" in param["augment.methods"]:
            transforms_list.append(
                TraceableTransform(
                    RandAffine(
                        prob=1.0, translate_range=(0.1, 0.1, 0.1), padding_mode="zeros"
                    )
                )
            )

        if "rotate" in param["augment.methods"]:
            transforms_list.append(
                TraceableTransform(
                    RandRotate(
                        range_x=90,  # Rotate by up to ±30 degrees around the x-axis (cranio-caudal)
                        range_y=0,  # No rotation around the y-axis
                        range_z=0,  # No rotation around the z-axis
                        prob=1.0,  # Apply with probability 1.0
                        padding_mode="zeros",
                    )
                )
            )

        if "flip.lr" in param["augment.methods"]:
            transforms_list.append(
                TraceableTransform(RandFlip(prob=1.0, spatial_axis=1))
            )

        if "flip.ud" in param["augment.methods"]:
            transforms_list.append(
                TraceableTransform(RandFlip(prob=1.0, spatial_axis=0))
            )

        if "noise" in param["augment.methods"]:
            transforms_list.append(
                TraceableTransform(RandGaussianNoise(prob=1.0, mean=0.0, std=0.1))
            )

        if "zoom" in param["augment.methods"]:
            transforms_list.append(
                TraceableTransform(RandZoom(prob=1.0, min_zoom=0.9, max_zoom=1.1))
            )

        if "intensity" in param["augment.methods"]:
            transforms_list.append(
                TraceableTransform(RandScaleIntensity(factors=0.1, prob=1.0))
            )
        # Other transforms...

        self.transforms_list = transforms_list
        self.param = param

        # Ensure self.__transform is always initialized
        if transforms_list:
            self.rand_some_of = RandSomeOf(transforms_list)
            self.__transform = self.rand_some_of  # Set the transform to RandSomeOf
        else:
            self.set_no_op_transform()  # Set a no-op transform        # Ensure rand_some_of is always initialized

    def set_no_op_transform(self):
        # Set self.rand_some_of to a no-op transform
        self.rand_some_of = self.no_op_transform

    @staticmethod
    def no_op_transform(img, num_transforms=None, seed=None):
        # No-op transform that simply returns the input unchanged
        return img

    def transform(self, input_data: ndarray, seed: int) -> ndarray:
        if self.__transform is None:
            return input_data

        # Determine the number of transforms to apply using the seed
        random.seed(seed)
        num_transforms = random.randint(
            self.param["augment.min"], self.param["augment.max"]
        )

        # Apply the selected number of transformations with the seed
        transformed_data = self.rand_some_of(
            img=input_data, num_transforms=num_transforms, seed=seed
        )
        # Get and print the trace of applied transformations
        return transformed_data

    def get_trace(self):
        """Retrieve the trace of applied transforms"""
        traces = []
        for transform in self.transforms_list:
            if isinstance(transform, TraceableTransform):
                if transform.trace:
                    traces.append(transform.trace)
            elif isinstance(transform, Compose):
                # If it's a Compose object, recursively get traces from its transforms
                for sub_transform in transform.transforms:
                    if (
                        isinstance(sub_transform, TraceableTransform)
                        and sub_transform.trace
                    ):
                        traces.append(sub_transform.trace)
        return traces
