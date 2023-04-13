from collections import OrderedDict
import numpy as np
import json
from ..base import DesignSpaceBase
from .key import MultiDimKey
import queue

__all__ = ["MultiDimSpace", "CategoricalSpace", "UniformCategoricalSpace"]


class DesignSpace(DesignSpaceBase):
    def __init__(self, policy):
        super().__init__()
        self._history = []
        self._history_file = None
        self._policy = policy

    def set_policy(self, policy):
        self._policy = policy

    def set_history_file(self, filename):
        self._history_file = filename

    def get_history_file(self):
        return self._history_file

    def record_history(self, key, config, value):
        self._history.append({"key": key, "config": config, "value": value})

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

    def get_next(self, need_best=False):
        raise NotImplementedError()


class MultiDimSpace(DesignSpace):
    def __init__(self, policy):
        super().__init__(policy)
        self._sub_spaces = OrderedDict()
        self._next_choices = queue.Queue()
        self._record_get_name = set()
        self._cur_choice = None
        self._constraint = None

    def prepare_next_choices(self, need_best=False):
        for i in range(5):
            self._next_choices.put(self.get_next(need_best))

    def get_next_for(self, name, need_best=False):
        if name in self._record_get_name:
            self._cur_choice = None
            self._record_get_name.clear()
        if self._cur_choice is None:
            if self._next_choices.empty():
                self.prepare_next_choices(need_best=need_best)
            self._cur_choice = self._next_choices.get()
        self._record_get_name.add(name)
        return self._cur_choice[0].children[name], self._cur_choice[1][name]

    def set_constraint(self, constraint):
        self._constraint = constraint

    def get_next(self, need_best=False):
        self._policy.set_inference_mode(need_best)
        return self._policy(self._sub_spaces, self._history)

    def add_subspace(self, key, subspace):
        self._cur_choice = None
        self._record_get_name.clear()
        self._next_choices = queue.Queue()
        self._policy.clear_cache()
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
        # key = [key] if not isinstance(key, (list, tuple)) else key
        assert len(key) == len(
            self), f"{key} of length {len(key)} vs {len(self)}"
        config = {}
        for k, v in key.children.items():
            assert isinstance(v, MultiDimKey)
            config[k] = self._sub_spaces[k][v]
        return config

    def __setitem__(self, key, value):
        config = self[key]
        self.record_history(key, config, value)

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
    def __init__(self, choices, policy):
        super().__init__(policy)
        assert isinstance(choices, (list, tuple))
        self._choices = list(choices)
        self._next_choice_keys = []

    def add_next_choice_keys(self, keys):
        self._next_choice_keys.extend(keys)

    def clear_next_choice_keys(self):
        self._next_choice_keys.clear()

    def get_next(self, need_best=False):
        self._policy.set_inference_mode(need_best)
        return self._policy(self._choices, self._history)

    def __len__(self):
        return len(self._choices)

    def __getitem__(self, key):
        assert isinstance(key, MultiDimKey)
        assert key.is_int_key() and key.value < len(self)
        return self._choices[key.value]

    def append(self, value):
        self._choices.append(value)

    def extend(self, lst):
        self._choices.extend(lst)

    def __contains__(self, value):
        return value in self._choices


class UniformCategoricalSpace(CategoricalSpace):
    def __init__(self, choices, policy):
        super().__init__(choices, policy=policy)
        if len(self._choices):
            type_cls = type(self._choices[0])
            for c in self._choices:
                if type(c) != type_cls:
                    raise ValueError(
                        f"{self.__class__} expects the same type for every choice.")
