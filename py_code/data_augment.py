import imgaug as ia
from custom import Dict, List
from imgaug import augmenters as iaa
from numpy import ndarray
from str_lib import (
    AUGMENT_MAX,
    AUGMENT_METHODS,
    AUGMENT_MIN,
    AUGMENT_PCT,
    ELASTIC,
    FLIP_LR,
    FLIP_UD,
    ROTATE,
    SCALE,
    TRANSLATE,
)


class DataAugmentation:
    def __init__(self, param: Dict = None):
        # = None means no data augmentation
        self.__transform = None

        # no augmentation parameter
        if param is None:
            return

        # no augmentation needed
        if param[AUGMENT_PCT] <= 0 or param[AUGMENT_METHODS] == []:
            return

        # augmentation needed
        # convert self.__transform to list to save values
        self.__transform = List()

        augment_dict = Dict()
        # iaa.ElasticTransformation():
        # alpha controls the strength of the displacement:
        # higher alpha mean that pixels are moved further.
        # sigma controls the smoothness of the displacement:
        # higher sigma lead to smoother patterns
        augment_dict[ELASTIC] = iaa.ElasticTransformation(alpha=(50.0, 70.0), sigma=9.0)
        augment_dict[SCALE] = iaa.Affine(scale={"x": (0.75, 1.33), "y": (0.75, 1.33)})
        augment_dict[TRANSLATE] = iaa.Affine(
            translate_percent={"x": (-0.25, 0.25), "y": (-0.25, 0.25)}
        )
        augment_dict[ROTATE] = iaa.Affine(rotate=(0, 360))
        augment_dict[FLIP_LR] = iaa.Fliplr(1.0)
        augment_dict[FLIP_UD] = iaa.Flipud(1.0)

        for i in augment_dict.keys():
            if i in param[AUGMENT_METHODS]:
                self.__transform.append(augment_dict[i])

        self.__transform = iaa.SomeOf(
            (param[AUGMENT_MIN], param[AUGMENT_MAX]),
            self.__transform,
            random_order=True,
        )
        self.__transform = iaa.Sometimes(param[AUGMENT_PCT], self.__transform)

    def transform(self, input_data: ndarray, seed: int) -> ndarray:
        if self.__transform is None:
            return input_data

        if len(input_data.shape) == 3:
            for i in range(input_data.shape[0]):
                ia.seed(seed)
                input_data[i] = self.__transform(images=input_data[i])
            return input_data

        elif len(input_data.shape) == 2:
            ia.seed(seed)
            return self.__transform(images=input_data)

        else:
            return input_data
