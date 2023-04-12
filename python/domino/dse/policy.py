import numpy as np
from ..base import PolicyBase
from .key import MultiDimKey


__all__ = ["CallablePolicy", "RandomPolicy"]


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


class RandomPolicy(CallablePolicy):
    def __call__(self, choices, history, *args):
        assert isinstance(choices, (list, tuple))
        assert len(choices) > 0
        assert isinstance(history, (list, tuple))
        if not self._inference_mode:
            idx = np.random.choice(range(len(choices)))
            return choices[idx]
        else:
            if len(history):
                best_h = history[0]
                for h in history:
                    if h["value"] > best_h:
                        best_h = h
                return best_h["config"]
            if self._fallback_choice is not None:
                assert type(choices[0]) == type(self._fallback_choice)
                return self._fallback_choice
            if self._fallback_choice_key is not None:
                assert self._fallback_choice_key.is_int_key()
                return choices[self._fallback_choice_key.first]
            raise RuntimeError(
                f"{self.__class__} has no history or fallback choice.")
