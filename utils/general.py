"""
General tiny help routines.
"""
from __future__ import absolute_import, division, print_function
import numpy as np
import re


def unpack_list_nd(input_, item_type=None, keep_types=[str, np.ndarray]):
    """unpack list of multi dimension into one dimension.
    """
    result = []
    for list_maybe in input_:
        is_list = hasattr(list_maybe, '__iter__')
        for keep in keep_types:
            if isinstance(list_maybe, keep):
                is_list = False
        if  is_list:
            result.extend(unpack_list_nd(list_maybe, item_type))
        else:
            if item_type is None or type(list_maybe) is item_type:
                result.append(list_maybe)
    return result

def errmsg(got, required, msg=""):
    return msg + "got: {0}, required: {1}.".format(got, required)


def seperate_file_name(file):
    """Analysis standard file name.
    """
    pfn = re.compile('\D+\d+\w*')
    if not pfn.match(file):
        raise ValueError('Invalid file name.')
    pid = re.compile('\D\d+')
    psu = re.compile('.\w*')
    mid = pid.search(file)
    prefix = file[:mid.start() + 1]

    id = int(file[mid.start() + 1:mid.end()])

    suffix = file[mid.end() + 1:]

    if suffix != '':
        seperater = file[mid.end()]
        if seperater != '.':
            raise ValueError('Invalid file name, seperater should be "."')
    return prefix, id, suffix


def form_file_name(prefix, id, suffix):
    """Standard file name of datas, given prefix and suffix.
    """
    filename = prefix + '%09d' % id + '.' + suffix
    return filename

# def valid_filename_form():
#     output = r"\w+\d[16].\w"
#     return
