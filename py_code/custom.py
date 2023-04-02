import copy
import json
import random
from natsort import natsorted
from typing import Union


def set_range(value, limit: tuple):
    low_limit = limit[0]
    up_limit = limit[1]
    if low_limit is not None:
        if value < low_limit:
            value = low_limit
    if up_limit is not None:
        if value > up_limit:
            value = up_limit
    return value


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

    def keys(self):
        return list(super().keys())

    def key_with_max_value(self):
        return max(super().keys(), key=(lambda k: self[k]))

    def key_with_min_value(self):
        return min(super().keys(), key=(lambda k: self[k]))


class List(list):
    def __init__(self, *args):
        # str to list, split with comma: "1,2,3,4" -> ["1","2","3","4"]
        if len(args) == 1 and isinstance(args[0], str):
            if args[0] == "":
                super().__init__()
            else:
                super().__init__(args[0].split(","))
        # normal init
        else:
            super().__init__(*args)

    # [1,2,3,4] -> "1,2,3,4"
    def to_str(self) -> str:
        return ",".join(str(i) for i in self)

    def copy(self):
        return copy.deepcopy(self)

    def find_identical_items(self, other_list: list):
        identical_items = set(self) & set(list(other_list))
        return List(identical_items)

    def shuffle(self, seed: int = None):
        # sort before shuffle, ensure to get specific results using specific seed
        self.sort()
        if seed is not None:
            random_state = random.getstate()
            random.seed(seed)
            random.shuffle(self)
            random.setstate(random_state)
        else:
            random.shuffle(self)

    def sort(self):
        super().__init__(natsorted(self))


class Json:
    def save(data: dict, path: str):
        with open(path, mode="w", encoding="utf-8") as json_file:
            # ensure_ascii == false, non-ASCII characters is available
            # skipkeys=True keys are not str will be skipped
            json.dump(
                data,
                json_file,
                ensure_ascii=False,
                indent=4,
                sort_keys=True,
                skipkeys=False,
            )

    # after json loaded, key(int) will become string
    def load(path: str) -> Dict:
        with open(path, mode="r") as json_file:
            data = json.load(json_file)
        data = Dict(data)
        # call "Json.save" to sort key
        Json.save(data=data, path=path)
        return data
