""" Option class 
Support get options from (in priority):
CLI
Arguments
JSON files
Default values
"""


# TODO: Implementation of json reader, herachy support.


class OptionBase:
    """ base option class """
    _name = 'Option'
    _settings = dict()

    def __init__(self):
        pass

    def _update_settings(self):
        """ update settings from vars to _settings, should not be called for changing options """
        all_vars = vars(self)
        for vname in all_vars:
            if vname.startswith('__'):
                continue
            if getattr(self, vname) is not None or self._settings.get(vname) is None:
                self._settings.update({vname: getattr(self, vname)})

    def update_attr(self):
        """ update settings from _settings to vars """
        for key, value in self._settings.items():
            if not hasattr(self, key) or value is not None:
                setattr(self, key, value)

    def update(self, field_name, value=None, is_priority=False, is_force=True):
        """ update settings with possible multiple possible values
        Args:
            field_name: filed name
            value: value, maybe a list of values
            is_priority: treat value as a list of values, find first arguments which is not None
            is_force: allow None value to overwrite a not None value
        Raises:
            ValueError if is_priority is True and value is not a list or tuple.
        """
        if is_priority:
            for v in value:
                if v is not None:
                    value_true = v
                    break
        else:
            value_true = value
        if self._settings.get(field_name) is None or value_true is not None or is_force:
            self._settings.update({field_name: value})
