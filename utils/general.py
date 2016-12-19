"""
General tiny help routines.
"""
import os
import re
import json
import logging
import numpy as np


def unpack_list_nd(input_, item_type=None, keep_types=(str, np.ndarray)):
    """unpack list of multi dimension into one dimension.
    :param input_: a container type object, will be expand if hasattr '__iter__'
    :param item_type: if not None, none item_type elements will be droped
    :param keep_types: do not expand these types.
    :returns: a list of elements
    """
    result = []
    for list_maybe in input_:
        is_list = hasattr(list_maybe, '__iter__')
        for keep in keep_types:
            if isinstance(list_maybe, keep):
                is_list = False
        if is_list:
            result.extend(unpack_list_nd(list_maybe, item_type))
        else:
            if item_type is None or isinstance(list_maybe, item_type):
                result.append(list_maybe)
    return result


def errmsg(got, required, msg=""):
    """Warp for error message.
    """
    return msg + "got: {0}, required: {1}.".format(got, required)


def seperate_file_name(file):
    """Analysis standard file name.
    """
    pat = re.compile(r'(\D*)(\d+)\.{0,1}(\w*)')
    match = pat.match(file)
    if not match:
        return None, None, None
    prefix = match.group(1)
    id_ = int(match.group(2))
    suffix = match.group(3)
    return prefix, id_, suffix


def form_file_name(prefix, id_, suffix=None):
    """Standard file name of datas for xlearn, given prefix and suffix.
    :param prefix: file name prefix
    :param id_: case if of data
    :param suffix: suffix of data, default = None
    :returns: formed filename
    """
    if suffix == '' or suffix is None:
        filename = prefix + '%09d' % id_
    else:
        filename = prefix + '%09d' % id_ + '.' + suffix
    return filename


def label_name(data_name, case_digit=None, label_prefix=None):
    """Get filename of label to a data name.
    :param data_name:   data filename
    :param case_digit:  number of digits for case. default = None
                        e.g. dataSSSDDDDDD.xxx will share label000DDDDDD.xxx as
                        label file with case_digit = 6.
    :param label_prefix:prefix of label filename. default is same prefix as
                        data file.
    :returns:            label filename
    """
    prefix, id_, suffix = seperate_file_name(data_name)
    if case_digit is not None:
        id_ = str(id_)
        id_ = id_[:-case_digit]
        id_ = int(id_)
    if label_prefix is None:
        label_prefix = prefix
    output = form_file_name(label_prefix, id, suffix)
    return output

# def valid_filename_form():
#     output = r"\w+\d[16].\w"
#     return


def merge_settings(settings=None, filenames=None, default_settings=None, **kwargs):
    """Merge settings from multiple file and args into one
    """
    logging.getLogger(__name__).debug("filenames:{}".format(filenames))
    if settings is None:
        settings = {}

    if default_settings is not None:
        settings.update(default_settings)

    if filenames is None:
        filenames = ()    
    if not isinstance(filenames, (list, tuple)):
        filenames = (filenames,)    
    for filename in filenames:
        filename = os.path.abspath(filename)
        with open(filename, 'r') as file_conf:
            tmp = json.load(file_conf)
            tmp = dict(filter(lambda x: x[1] is not None, tmp.items()))
            settings.update(tmp)

    filted_kwargs = dict(filter(lambda x: x[1] is not None, kwargs.items()))

    settings.update(filted_kwargs)

    return settings
