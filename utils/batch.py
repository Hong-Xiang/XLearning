import pathlib
import json
from pprint import pprint
from collections import ChainMap

class Project:    
    def __init__(self, name, path, options=None, option_files=None):
        self.name = name
        self.path = pathlib.Path(path)
        option_dicts = []                
        if isinstance(options, dict):
            option_dicts = [options]
        if option_files:
            for fn in option_files:
                option_dicts.append(json.load(path/option_files))
        self.options = ChainMap(option_dicts)
    
    def print_options(self):
        pprint(self.options)
    
    def run(self):
        pass

def option_product(projects, key, values):
    """
    Inputs: 
    
    projects: list of projects

    key: option key

    values:

    Returns:
        A list of projects with given options
    """
    projects_lists = []
    for p in projects:
        projects.append(Project())
