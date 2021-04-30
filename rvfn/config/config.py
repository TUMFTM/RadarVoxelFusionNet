from abc import ABC, abstractmethod
from copy import deepcopy
import json


class Config(ABC):
    """
    Abstract class for configuration. This provides functionality for reading
     configs from dictionaries with dynamic type-checking, and automatic
     compositions of sub-configs.
    To create a config class inherit from this and implement get_default_conf()
     to return a dictionary with the default key-value config pairs. The values
     may themselves be Config sub-types, in which case users can also pass a
     dict for that key and a new instance of the sub-type will be created
     automatically.
    All the elements in the dict will be added as attributes to be used
     instead of reading the dict directly.
    You may also implement a verify() function that return false if there is a
     problem with the config. In which case the constructor will raise an
     exception.
    """

    def __init__(self, conf: dict = None):
        """
        Create an instance using conf to update default parameters.
        Args:
            conf: dict used to replace the default parameters. Keys in this
             dict must be present in the dict returned by get_default_dict()
             but not all keys need to be provided. The value for keys that
             are not present will be set to default.
             The values will be type-checked against the default dict. If the
              default type is tuple, you may also pass a list instead (this is
              for json compatibility)
             If the type of a certain config is a Config instance itself, you
              may also pass a dict instead.
        """
        super().__init__()

        self._config = self.get_default_conf()
        if conf:
            for key in conf:
                # Complain if key not in default configs
                # The user has probably made a typo
                if key not in self._config:
                    raise KeyError('Key not recognized: ' + key)

                # Type-check the input value, and construct new sub-config
                #  instances if necessary
                if self._config[key] is not None and conf[key] is not None:
                    # If a value's default is a Config instance itself, and the
                    #  corresponding input type is a dict, use it to create a
                    #  new Config object of the correct type.
                    if isinstance(self._config[key], Config) and isinstance(
                            conf[key], dict):
                        conf[key] = self._config[key].__class__(conf[key])

                    # Complain if type of a field doesn't match the
                    #  corresponding type in the default dict
                    if type(self._config[key]) != type(conf[key]):
                        # Allow providing list instead of tuple to get around
                        #  lack of tuple types in json
                        if isinstance(self._config[key], tuple) and isinstance(
                                conf[key], list):
                            pass
                        else:
                            raise ValueError(
                                'Type of input config does not match default '
                                'type.\n' + 'For key: ' + key + ', Expected: '
                                + str(type(self._config[key])) + ', Got: ' +
                                str(type(conf[key])))

                self._config[key] = conf[key]

        self.process_params()

        if not self.verify():
            raise ValueError('Configuration is invalid')

    @classmethod
    @abstractmethod
    def get_default_conf(cls) -> dict:
        """
        Overload this and return the default config dict
        """
        pass

    def verify(self):
        """
        Optionally overload to implement config verification
        """
        return True

    def process_params(self):
        """
        Optionally overload to do any necessary changes to the parameters
        """
        pass

    def dict(self):
        """
        Recursively converts all the config and all subconfigs to dicts
        Returns: a dict representing the config

        """
        ret = self._config.copy()
        for key, val in ret.items():
            if isinstance(val, Config):
                ret[key] = val.dict()

        return ret

    @classmethod
    def from_file(cls, file_path):
        """
        Create config from json file
        """
        with open(file_path) as file:
            conf_dict = json.load(file)
        return cls(conf_dict)

    def __contains__(self, item):
        return item in self._config

    def __getitem__(self, item):
        return self._config[item]

    def __setitem__(self, key, value):
        self._config[key] = value

    def __getattr__(self, item):
        if item not in self.__dict__ and item in self.__getattribute__(
                '_config'):
            return self[item]
        raise AttributeError(item)

    def __setattr__(self, key, value):
        if key == '_config':
            super().__setattr__(key, value)
        elif key in self.__getattribute__('_config'):
            self[key] = value
        else:
            raise AttributeError(key)

    def copy(self):
        return deepcopy(self)
