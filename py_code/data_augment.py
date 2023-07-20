import imgaug as ia
from custom import Dict, List
from imgaug import augmenters as iaa
from numpy import ndarray


class DataAugmentation:
    def __init__(self, param: Dict = None):
        # = None means no data augmentation
        self.__transform = None

        # no augmentation parameter
        if param is None:
            return

        # no augmentation needed
        if param["pct"] <= 0 or param["methods"] == []:
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
        augment_dict["elastic"] = iaa.ElasticTransformation(
            alpha=(50.0, 70.0), sigma=9.0
        )
        augment_dict["scale"] = iaa.Affine(scale={"x": (0.75, 1.33), "y": (0.75, 1.33)})
        augment_dict["translate"] = iaa.Affine(
            translate_percent={"x": (-0.25, 0.25), "y": (-0.25, 0.25)}
        )
        augment_dict["rotate"] = iaa.Affine(rotate=(0, 360))
        augment_dict["flip.lr"] = iaa.Fliplr(1.0)
        augment_dict["flip.ud"] = iaa.Flipud(1.0)

        for i in augment_dict.keys():
            if i in param["methods"]:
                self.__transform.append(augment_dict[i])

        self.__transform = iaa.SomeOf(
            (param["min"], param["max"]), self.__transform, random_order=True
        )
        self.__transform = iaa.Sometimes(param["pct"], self.__transform)

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
