""" Option class
Support get options from (in priority):
CLI
Arguments
JSON files
Default values
"""


# TODO: Implementation of json reader, herachy support.

import json
import click
from pathlib import Path


class Params(dict):

    class ShortCut:
        pass

    def __init__(self, *args, **kwargs):
        super(Params, self).__init__(*args, **kwargs)
        self.__short_cut = Params.ShortCut()

    def update_short_cut(self):
        for k in self:
            setattr(self.__short_cut, k, self[k])
    
    def add(name, value):
        self[name] = value
        setattr(self.__short_cut, name, value)

    @property
    def short_cut(self):
        return self.__short_cut


# def standarize_configs(configs):
#     """ standarize_configs to list of pathlib.Path objects """
#     if configs is not None:
#         if not isinstance(configs, (list, tuple)):
#             raise TypeError(
#                 'configs must be a list of str or pathlib.Path.')
#         configs_std = []
#         for cfg in configs:
#             configs_std.append(pathlib.Path(cfg))
#     else:
#         configs_std = None
#     return configs_std


# class Options(dict):
#     """ base class for Options
#         Options is a dict of options or paremeters.
#         Each parameters has a id and name.
#     """

#     def __init__(self, configs=None, **kwargs):
#         """ Constructer of Options.
#         Args:
#             configs: list of pathlib.Path object or list of str, paths of .JSON config files.
#         """
#         super(Options, self).__init__(**kwargs)
#         configs_std = standarize_configs(configs)
#         for cfg in configs_std:
#             self.load(cfg)
#         valid_dict = self._get_valid(**kwargs)
#         self.update(valid_dict)

#     def _is_valid(self, key_name):
#         return False

#     def load(self, filepath):
#         """ load a JSON config file from disk """
#         raw_config = json.load(str(filepath))
#         for k, v in raw_config.items():
#             valid_config = self._get_valid(k, v)
#         self.update(valid_config)

#     def save(self, filepath, is_expand=True):
#         """ save options to a JSON file
#         Args:
#             filepath: str or pathlib.Path object, path of .JSON file to save,
#             is_iterative: flag of whether save child options
#         """
#         output_dict = {}
#         if isinstance(filepath, str):
#             filepath = Path(filepath)
#         for v in vars(self):
#             if v.startswith('__'):
#                 continue
#             atr = getattr(self, v)
#             if isinstance(atr, Options):
#                 if is_expand:
#                     output_dict[v] = atr
#                 else:
#                     pass
#         with open(str(filepath), 'w') as f:
#             f.write(json.dumps(output_dict, separators=[' :', ','], indent=2))

    # def _update_settings(self):
    #     """ update settings from vars to _settings, should not be called for changing options """
    #     all_vars = vars(self)
    #     for vname in all_vars:
    #         if vname.startswith('__'):
    #             continue
    #         if getattr(self, vname) is not None or self._settings.get(vname) is None:
    #             self._settings.update({vname: getattr(self, vname)})

    # def update_attr(self):
    #     """ update settings from _settings to vars """
    #     for key, value in self._settings.items():
    #         if not hasattr(self, key) or value is not None:
    #             setattr(self, key, value)

    # def update(self, field_name, value=None, is_priority=False, is_force=True):
    #     """ update settings with possible multiple possible values
    #     Args:
    #         field_name: filed name
    #         value: value, maybe a list of values
    #         is_priority: treat value as a list of values, find first arguments which is not None
    #         is_force: allow None value to overwrite a not None value
    #     Raises:
    #         ValueError if is_priority is True and value is not a list or tuple.
    #     """
    #     if is_priority:
    #         for v in value:
    #             if v is not None:
    #                 value_true = v
    #                 break
    #     else:
    #         value_true = value
    #     if self._settings.get(field_name) is None or value_true is not None or is_force:
    #         self._settings.update({field_name: value})
