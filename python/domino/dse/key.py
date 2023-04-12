

__all__ = ["MultiDimKey"]

class MultiDimKey:
    def __init__(self, first, second=None):
        assert second is None or isinstance(second, (list, tuple))
        self.first = first
        self.second = second
        if self.second is not None:
            for it in self.second:
                assert isinstance(it, MultiDimKey)
                
    def is_int_key(self):
        return isinstance(self.first, int) and self.second is None

    @staticmethod
    def make_multi_dim_key(k):
        if isinstance(k, MultiDimKey):
            return k
        elif isinstance(k, int):
            return MultiDimKey(k)
        elif isinstance(k, (list, tuple)):
            return [MultiDimKey.make_multi_dim_key(kk) for kk in k]
        elif isinstance(k, dict):
            ret = []
            for kk, vv in k.items():
                vvk = MultiDimKey.make_multi_dim_key(vv)
                ret.append(MultiDimKey(kk, vvk))
            return ret
        else:
            raise ValueError(f"Can't convert {k} to MultiDimKey.")