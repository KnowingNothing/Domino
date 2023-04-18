from builtins import callable
from functools import lru_cache
import math
from ..space import MultiDimSpace, UniformCategoricalSpace

__all__ = ["DimSplitSpace", "MultiDimTileSpace"]


@lru_cache
def get_all_factors(value: int):
    end = int(math.sqrt(value))
    ret = []
    for i in range(1, end+1):
        if value % i == 0:
            ret.append(i)
            ret.append(value // i)
    return list(sorted(ret))


@lru_cache
def split_to_factors(value: int, parts: int):
    factors = get_all_factors(value)
    ret = []

    def helper(cur_id, cur, left):
        nonlocal ret
        if cur_id == parts - 1:
            ret.append(cur + [left])
            return
        else:
            for f in factors:
                if left % f == 0:
                    helper(cur_id + 1, cur + [f], left//f)
    helper(0, [], value)
    return ret


class DimSplitSpace(UniformCategoricalSpace):
    def __init__(self, length, nparts, policy, constraints=None):
        assert isinstance(length, int), "Only support static shape"
        assert isinstance(nparts, int) and nparts > 0
        assert length > 0, length
        constraints = [] if constraints is None else constraints
        assert isinstance(constraints, (list, tuple))
        for c in constraints:
            assert callable(c), "Expect callable function as constraint"
        choices = split_to_factors(length, nparts)
        filtered_choices = []
        for c in choices:
            valid = True
            for f in constraints:
                if not f(c):
                    valid = False
                    break
            if valid:
                filtered_choices.append(c)
        super().__init__(filtered_choices, policy=policy)


class MultiDimTileSpace(MultiDimSpace):
    def __init__(self, dims):
        assert isinstance(dims, dict)
        super().__init__()
        for name, value in dims.items():
            assert isinstance(name, str)
            assert isinstance(value, tuple) and len(value) == 3
            length, nparts, constraints = value
            self[name] = DimSplitSpace(
                length, nparts=nparts, constraints=constraints)
