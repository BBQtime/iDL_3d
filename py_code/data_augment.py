import imgaug as ia
from imgaug import augmenters as iaa
from numpy import ndarray


class DataAugment:
    def __init__(
        self,
        pct: float = 0.0,
        method: str = None,
        low_limit: float = 0,
        up_limit: float = 0,
    ):
        self.__pct = pct
        self.__method = method

        if up_limit < low_limit:
            up_limit = low_limit

        if self.__pct > 1.0:
            self.__pct = 1.0

        if self.__pct <= 0.0 or self.__method is None:
            return

        # iaa.ElasticTransformation():
        # alpha controls the strength of the displacement:
        # higher alpha mean that pixels are moved further.
        # sigma controls the smoothness of the displacement:
        # higher sigma lead to smoother patterns
        elastic = iaa.ElasticTransformation(alpha=(50.0, 70.0), sigma=9.0)

        translate = iaa.Affine(
            translate_percent={"x": (-0.25, 0.25), "y": (-0.25, 0.25)}
        )
        rotate = iaa.Affine(rotate=(-30, 30))
        scale = iaa.Affine(scale={"x": (0.75, 1.33), "y": (0.75, 1.33)})
        combine = iaa.SomeOf((low_limit, up_limit), [translate, elastic, rotate, scale])

        if self.__method == "scale":
            self.__method = scale
        elif self.__method == "translate":
            self.__method = translate
        elif self.__method == "rotate":
            self.__method = rotate
        elif self.__method == "elastic":
            self.__method = elastic
        else:
            self.__method = combine

        self.__transform = iaa.Sometimes(self.__pct, self.__method)

    def run(self, input_data: ndarray, seed: int) -> ndarray:
        if self.__pct <= 0.0 or self.__method is None:
            return input_data
        ia.seed(seed)
        output_data = self.__transform(images=input_data)
        return output_data
