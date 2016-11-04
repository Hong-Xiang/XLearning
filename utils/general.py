"""
General tiny help routines.
"""
from __future__ import absolute_import, division, print_function
import numpy as np
def unpack_list(input_, item_type=None, keep_types=[str, np.ndarray]):
    result = []    
    for list_maybe in input_:
        is_list = hasattr(list_maybe, '__iter__')
        for keep in keep_types:
            if isinstance(list_maybe, keep):
                is_list = False
        if  is_list:
            result.extend(unpack_list(list_maybe, item_type))            
        else:
            if item_type is None or type(list_maybe) is item_type:                
                result.append(list_maybe)
    return result