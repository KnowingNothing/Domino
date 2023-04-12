from collections import OrderedDict
import numpy as np
import json
from ..base import DesignSpaceBase
from .key import MultiDimKey

__all__ = ["MultiDimSpace", "CategoricalSpace", "UniformCategoricalSpace"]


class DesignSpace(DesignSpaceBase):
    def __init__(self):
        super().__init__()
        self._history = []
        self._history_file = None

    def set_history_file(self, filename):
        self._history_file = filename

    def get_history_file(self):
        return self._history_file

    def record_history(self, config, value):
        self._history.append({"config": config, "value": value})

    def save_to_file(self, filename=None):
        assert filename or self._history_file
        filename = filename if filename is not None else self._history_file
        assert isinstance(filename, str)
        with open(filename, "a") as fout:
            for line in self._history:
                string = json.dumps(line)
                fout.write(string + "\n")

    def load_from_file(self, filename=None):
        self._history.clear()
        assert filename or self._history_file
        filename = filename if filename is not None else self._history_file
        assert isinstance(filename, str)
        with open(filename, "r") as fin:
            for line in fin:
                obj = json.loads(line)
                self._history.append(obj)

    def append_from_file(self, filename=None):
        assert filename or self._history_file
        filename = filename if filename is not None else self._history_file
        assert isinstance(filename, str)
        with open(filename, "r") as fin:
            for line in fin:
                obj = json.loads(line)
                self._history.append(obj)

    def get_next(self, policy):
        raise NotImplementedError()


class MultiDimSpace(DesignSpace):
    def __init__(self):
        super().__init__()
        self._sub_spaces = OrderedDict()

    def add_subspace(self, key, subspace):
        assert isinstance(
            subspace, DesignSpace), f"Can't treat {subspace} as subspace."
        self._sub_spaces[key] = subspace

    def get_subspace(self, key):
        assert key in self._sub_spaces, f"{key} is not a name of subspace."
        return self._sub_spaces[key]

    def del_subspace(self, key):
        del self._sub_spaces[key]

    def has_subspace(self, key):
        return key in self._sub_spaces

    def __getitem__(self, key):
        key = [key] if not isinstance(key, (list, tuple)) else key
        assert len(key) == len(self)
        config = {}
        for k in key:
            assert isinstance(k, MultiDimKey)
            config[k.first] = self._sub_spaces[k.second]
        return config

    def __setitem__(self, key, value):
        config = self[key]
        self.record_history(config, value)

    def __contains__(self, key):
        try:
            config = self[key]
            return True
        except Exception as e:
            return False

    def __len__(self):
        return len(self._sub_spaces)

    def keys(self):
        return self._sub_spaces.keys()

    def items(self):
        return self._sub_spaces.items()


class CategoricalSpace(DesignSpace):
    def __init__(self, choices):
        super().__init__()
        assert isinstance(choices, (list, tuple))
        self._choices = list(choices)

    def get_next(self, policy):
        assert callable(policy)
        return policy(self._choices, self._history)

    def __len__(self):
        return len(self._choices)

    def __getitem__(self, key):
        assert isinstance(key, MultiDimKey)
        assert key.is_int_key() and key.first < len(self)
        return self._choices[key.first]

    def append(self, value):
        self._choices.append(value)

    def extend(self, lst):
        self._choices.extend(lst)

    def __contains__(self, value):
        return value in self._choices


class UniformCategoricalSpace(CategoricalSpace):
    def __init__(self, choices):
        super().__init__(choices)
        if len(self._choices):
            type_cls = type(self._choices[0])
            for c in self._choices:
                if type(c) != type_cls:
                    raise ValueError(
                        f"{self.__class__} expects the same type for every choice.")
