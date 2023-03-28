import numpy as np
import copy
from typing import Union


# nested dictionary
# after json loaded, key(int type) will transfrom into string
# better make all keys "string" type
class Dict(dict):
    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value

    def copy(self):
        return copy.deepcopy(self)

    def __convert_data_type(self, data: Union[dict, list]):
        if isinstance(data, dict):
            for key in data:
                if isinstance(data[key], bool):
                    pass
                elif isinstance(data[key], dict):
                    self.__convert_data_type(data[key])
                elif isinstance(data[key], list):
                    self.__convert_data_type(data[key])
                elif isinstance(data[key], int):
                    data[key] = Int(data[key])
                elif isinstance(data[key], float):
                    data[key] = Float(data[key])
                elif isinstance(data[key], str):
                    data[key] = Str(data[key])

        elif isinstance(data, list):
            for i in range(len(data)):
                if isinstance(data[i], bool):
                    pass
                elif isinstance(data[i], dict):
                    self.__convert_data_type(data[i])
                elif isinstance(data[i], list):
                    self.__convert_data_type(data[i])
                elif isinstance(data[i], int):
                    data[i] = Int(data[i])
                elif isinstance(data[i], float):
                    data[i] = Float(data[i])
                elif isinstance(data[i], str):
                    data[i] = Str(data[i])

    def __revert_data_type(self, data: Union[dict, list]):
        if isinstance(data, dict):
            for key in data:
                if isinstance(data[key], bool):
                    pass
                elif isinstance(data[key], dict):
                    self.__revert_data_type(data[key])
                elif isinstance(data[key], list):
                    self.__revert_data_type(data[key])
                elif isinstance(data[key], Int):
                    data[key] = int(data[key])
                elif isinstance(data[key], Float):
                    data[key] = float(data[key])
                elif isinstance(data[key], Str):
                    data[key] = str(data[key])

        elif isinstance(data, list):
            for i in range(len(data)):
                if isinstance(data[i], bool):
                    pass
                elif isinstance(data[i], dict):
                    self.__revert_data_type(data[i])
                elif isinstance(data[i], list):
                    self.__revert_data_type(data[i])
                elif isinstance(data[i], Int):
                    data[i] = int(data[i])
                elif isinstance(data[i], Float):
                    data[i] = float(data[i])
                elif isinstance(data[i], Str):
                    data[i] = str(data[i])

    # Change int/float/str/list into Int/Float/Str/List
    def convert_data_type(self):
        self.__convert_data_type(self)

    # Change Int/Float/Str/List into int/float/str/list
    def revert_data_type(self):
        self.__revert_data_type(self)


class List(list):
    pass


class Str(str):
    pass


class Float(np.ndarray):
    def __new__(cls, value: float):
        value = float(value)
        obj = np.asarray(value).view(cls)
        obj.info = None
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.info = getattr(obj, "info", None)

    def set_range(self, low_limit: float = None, up_limit: float = None):
        if low_limit is not None:
            low_limit = float(low_limit)
            if self < low_limit:
                self -= self
                self += low_limit

        if up_limit is not None:
            up_limit = float(up_limit)
            if self > up_limit:
                self -= self
                self += up_limit


class Int(np.ndarray):
    def __new__(cls, value: int):
        if isinstance(value, float):
            value = round(value)
        else:
            value = int(value)
        obj = np.asarray(value).view(cls)
        obj.info = None
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.info = getattr(obj, "info", None)

    def set_range(self, low_limit: int = None, up_limit: int = None):
        if low_limit is not None:
            low_limit = int(low_limit)
            if self < low_limit:
                self -= self
                self += low_limit

        if up_limit is not None:
            up_limit = int(up_limit)
            if self > up_limit:
                self -= self
                self += up_limit
