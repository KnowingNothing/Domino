import numpy as np
from ..base import PolicyBase
from .key import MultiDimKey
from .space import MultiDimSpace


__all__ = ["CallablePolicy", "AnnealingMutateOneDim",
           "CategoricalRandomPolicy"]


class CallablePolicy(PolicyBase):
    def __init__(self, fallback_choice=None, fallback_choice_key=None):
        super().__init__()
        self._inference_mode = False
        self._fallback_choice = fallback_choice
        self._fallback_choice_key = MultiDimKey.make_multi_dim_key(
            fallback_choice_key) if fallback_choice_key is not None else fallback_choice_key

    def enable_inference(self):
        self._inference_mode = True

    def disable_inference(self):
        self._inference_mode = False

    def set_inference_mode(self, value):
        self._inference_mode = value

    def __call__(self, *args):
        raise NotImplementedError()


class AnnealingMutateOneDim(CallablePolicy):
    def __init__(self, fallback_choice=None, fallback_choice_key=None):
        super().__init__(fallback_choice, fallback_choice_key)
        self._counter = 0
        self._visit = set()

    def clear_cache(self):
        self._visit = set()

    def find_start_point(self, history):
        if not len(history):
            return None
        best_v = -float("inf")
        best_h = None
        for h in history:
            if h["value"] > best_v:
                best_v = h["value"]
                best_h = h
        num_history = len(history)
        for i in range(100):
            idx = np.random.choice(range(num_history))
            h = history[idx]
            prob = np.exp(h["value"]-best_v)
            if np.random.random() < prob:
                return h
        assert best_h is not None
        return best_h

    def find_best_point(self, history):
        best_v = -float("inf")
        best_h = None
        for h in history:
            if h["value"] > best_v:
                best_v = h["value"]
                best_h = h
        assert best_h is not None
        return best_h

    def __call__(self, subspaces, history, *args):
        if self._inference_mode:
            return self.find_best_point(history)
        ret_key = MultiDimKey(None)
        ret_config = {}
        # the subspaces is an OrderedDict
        max_trial = 1000
        for trial in range(max_trial):
            i = 0
            start_h = self.find_start_point(history)
            if start_h is None:
                start_key = MultiDimKey(None)
                start_config = {}
                for name, subspace in subspaces.items():
                    key, config = subspace.get_next()
                    start_key.children[name] = key
                    start_config[name] = config
                    start_value = -float("inf")
            else:
                start_key, start_config, start_value = start_h["key"], start_h["config"], start_h["value"]

            for name, subspace in subspaces.items():
                if i == self._counter:
                    key, config = subspace.get_next()
                    ret_key.children[name] = key
                    ret_config[name] = config
                else:
                    ret_key.children[name] = start_key.children[name]
                    ret_config[name] = start_config[name]
                i += 1
            # update counter
            self._counter += 1
            if self._counter > len(subspaces):
                self._counter = 0
            if str(ret_key) not in self._visit:
                self._visit.add(str(ret_key))
                # print(f"[Policy] use start point {start_config} with value {start_value}")
                return ret_key, ret_config
        print("[Warning] Can't find new candidates.")
        return ret_key, ret_config


class CategoricalRandomPolicy(CallablePolicy):
    def __call__(self, choices, history, *args):
        assert isinstance(choices, (list, tuple))
        assert len(choices) > 0
        assert isinstance(history, (list, tuple))
        if not self._inference_mode:
            idx = np.random.choice(range(len(choices)))
            return MultiDimKey.make_multi_dim_key(idx), choices[idx]
        else:
            if len(history):
                best_h = history[0]
                for h in history:
                    if h["value"] > best_h:
                        best_h = h
                assert isinstance(
                    best_h["config"], MultiDimKey) and best_h["config"].is_int_key()
                return best_h["config"], choices[best_h["config"].value]
            if self._fallback_choice is not None:
                assert type(choices[0]) == type(self._fallback_choice)
                return MultiDimKey.make_multi_dim_key(-1), self._fallback_choice
            if self._fallback_choice_key is not None:
                assert self._fallback_choice_key.is_int_key()
                return self._fallback_choice_key, choices[self._fallback_choice_key.value]
            raise RuntimeError(
                f"{self.__class__} has no history or fallback choice.")
