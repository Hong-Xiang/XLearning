from collections import defaultdict

class DefaultDict(defaultdict):
    def __init__(self, *args, **kwargs):
        defaultdict(*args, **kwargs)
    
    def append(self, d):
        for k, v in d.items():
            tmp = self.get(k)
            if tmp is None:
                self[k] = []
            self[k].append(v)