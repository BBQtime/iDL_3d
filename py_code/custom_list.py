import copy
import itertools
import random

from natsort import natsorted


class List(list):
    # slicing operations on List still return an instance of List
    def __getitem__(self, key):
        result = super().__getitem__(key)
        if isinstance(key, slice):
            return List(result)
        else:
            return result

    def __init__(self, *args):
        # normal init
        if len(args) == 0:
            super().__init__(*args)
        elif len(args) == 1:
            # str to list, split with comma: "1,2,3,4" -> ["1","2","3","4"]
            if isinstance(args[0], str):
                if args[0] == "":
                    super().__init__()
                else:
                    super().__init__(args[0].split(","))
            # init int and float like List(1) or List(2.0)
            elif isinstance(args[0], int) or isinstance(args[0], float):
                super().__init__([args[0]])
            else:
                super().__init__(*args)
        # normal init
        else:
            super().__init__(*args)

    # [1,2,3,4] -> "1,2,3,4"
    def to_str(self) -> str:
        return ",".join(str(i) for i in self)

    def copy(self):
        return copy.deepcopy(self)

    def find_overlap(self, other_list: list):
        overlap = set(self) & set(other_list)
        overlap = List(overlap)
        overlap.shuffle()
        return overlap

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

    def sort(self, reverse: bool = False):
        super().__init__(natsorted(self, reverse=reverse))

    def remove_duplicates(self):
        self[:] = List(set(self))

    def get_combinations(self, length: int = 2):
        result = itertools.combinations(self, length)
        return List(result)
