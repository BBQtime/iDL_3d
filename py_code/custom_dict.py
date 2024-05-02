import copy

from custom_list import List


# nested dictionary
# (1) new_dict=Dict(origin_dict), change new_dict will not change origin_dict
# (2) better make all keys "str", because g.load_json() will change key type(int type) into string
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
        return List(super().keys())

    def key_with_max_value(self):
        return max(super().keys(), key=(lambda k: self[k]))

    def key_with_min_value(self):
        return min(super().keys(), key=(lambda k: self[k]))

    def sort_by_value(self, reverse: bool = False):
        sorted_dict = sorted(self.items(), key=lambda item: item[1], reverse=reverse)
        self.clear()
        self.update(sorted_dict)

    # {"0": [a, b], "1": [c, d], "2": [e]} -> [a, b, c, d, e]
    def to_list(self):
        output_list = List()
        for value in self.values():
            if isinstance(value, dict):
                sub_list = Dict(value).to_list()
                output_list.extend(sub_list)
            elif isinstance(value, list):
                output_list.extend(value)
            else:
                output_list.append(value)
        return output_list

    def key_index(self, key_name) -> int:
        idx = 0
        for i in self.keys():
            if i == key_name:
                return idx
            idx += 1
        return -1
