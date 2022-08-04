# nested dictionary
# after json loaded, key(int type) will transfrom into string
# better make all keys "string" type
class NestedDict(dict):
    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value
