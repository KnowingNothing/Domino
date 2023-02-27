import domino
import pytest
from domino.program_ir import *


def test_arch_memory_level():
    null = Evaluate(0)
    cur_level = ComputeLevel(0, null, [])
    for level in range(0, 5):
        cur_level = MemoryLevel(level, null, [cur_level])
    res = print_ir(cur_level, print_out=False)
    assert res


if __name__ == "__main__":
    test_arch_memory_level()
