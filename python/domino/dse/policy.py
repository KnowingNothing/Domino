import numpy as np
from ..base import PolicyBase


__all__ = ["CallablePolicy", "RandomPolicy"]


class CallablePolicy(PolicyBase):
    def __call__(self, *args):
        raise NotImplementedError()


class RandomPolicy(CallablePolicy):
    def __call__(self, choices, *args):
        assert isinstance(choices, (list, tuple))
        idx = np.random.choice(range(len(choices)))
        return choices[idx]
