import imgaug as ia
from imgaug import augmenters as iaa
from numpy import ndarray


class DataAugmentation:
    def __init__(
        self,
        methods: list = [],  # list of str
        pct: float = 0,
        low_limit: int = 0,
        up_limit: int = 0,
    ):
        self.__transform = None

        if pct <= 0 or methods == []:
            return
        if up_limit < low_limit:
            up_limit = low_limit
        if pct > 1:
            pct = 1

        augment_dict = dict()
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

        for i in ["elastic", "scale", "translate", "rotate", "flip.lr", "flip.ud"]:
            if i in methods:
                if self.__transform is None:
                    self.__transform = []
                self.__transform.append(augment_dict[i])

        self.__transform = iaa.SomeOf(
            (low_limit, up_limit), self.__transform, random_order=True
        )
        self.__transform = iaa.Sometimes(pct, self.__transform)

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
