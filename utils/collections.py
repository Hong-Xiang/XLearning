from collections import UserDict, defaultdict

class AttrDict(UserDict):
    CLI_DICT = dict()
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
    
    def __getattr__(self, key):
        value = self.data.get(key)
        if value is None:
            value = self.CLI_DICT.get(key)            
        return value
    
    def has_key(self, key):
        return key in self.data or key in self.CLI_DICT
    
    def set_default(self, key, value):
        if self.data.get(key) is None:
            self.data[key] = value

class ListDict(defaultdict):
    def __init__(self):
        super(ListDict, self).__init__(list)
    
    def append(self, new_dict):
        for k in new_dict:
            self[k].append(new_dict[k])

class DefaultDict(defaultdict):
    def __init__(self, *args, **kwargs):
        super(DefaultDict, self).__init__(*args, **kwargs)

    def append(self, new_dict):
        for k in new_dict:
            self[k].append(new_dict[k])
    # def __getattr__(self, key):
    #     value = self.data.get(key)      
    #     return value