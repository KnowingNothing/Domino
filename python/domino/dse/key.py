import numpy as np
import json

__all__ = ["MultiDimKey"]


class MultiDimKey:
    def __init__(self, value, children=None):
        children = {} if children is None else children
        assert isinstance(children, dict)
        self.value = value
        self.children = children
        for k, v in children.items():
            assert isinstance(v, MultiDimKey)

    def is_int_key(self):
        return isinstance(self.value, int) and len(self.children) == 0

    def to_json(self):
        return {
            "value": self.value,
            "children": {k: v.to_json() for k, v in self.children.items()}
        }

    def __str__(self):
        return json.dumps(self.to_json())

    def __repr__(self):
        return str(self)

    def __len__(self):
        return len(self.children)

    @staticmethod
    def make_multi_dim_key(k):
        if isinstance(k, MultiDimKey):
            return k
        elif isinstance(k, (int, np.int32, np.int64)):
            return MultiDimKey(int(k))
        elif isinstance(k, dict):
            ret = {}
            for kk, vv in k.items():
                vvk = MultiDimKey.make_multi_dim_key(vv)
                ret[kk] = vvk
            return MultiDimKey(None, ret)
        else:
            raise ValueError(f"Can't convert {type(k)}: {k} to MultiDimKey.")
