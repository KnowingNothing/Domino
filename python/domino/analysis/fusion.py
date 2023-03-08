from dominoc import ir, analysis
from ..program_ir import (
    ComputeLevel, MemoryLevel, Evaluate, Tensor, Var, Loop, Iterator, Range,
    make_prod_consum_graph, print_ir, simplify_expr, substitute_expr)
from ..passes import ProdConsumGraph
from typing import List, Dict
import queue
import math
from functools import lru_cache

__all__ = ["MemoryLevelTree", "create_tree", "generate_merged_memory_level_trees",
           "memory_level_tree_tiling", "infer_producer_shape",
           "generate_tile_tensor_computation_space"]


MemoryLevelTree = analysis.MemoryLevelTree

def create_tree(tensor: Tensor, levels: List[int]):
    if tensor.init is None:
        initial_bounds = {}
        return MemoryLevelTree(levels, tensor.var, initial_bounds)
    elif len(tensor.updates) == 0:
        compute = tensor.init
    elif len(tensor.updates) == 1:
        compute = tensor.updates[0]
    else:
        raise NotImplementedError("Computations with more than 1 update are not supported yet.")
    
    if tensor.init.has_reduce():
        raise RuntimeError("Don't konw how to handle reductions in init stages.")
    initial_bounds = {l.var: l.dom for l in compute.all_loops()}
    return MemoryLevelTree(levels, tensor.var, initial_bounds)


def generate_merged_memory_level_trees(root_tensor: Tensor, levels: List[int]):
    """
    Generate all possible merged memory level trees。
    Args:
    ---
    root_tensor:
        The final output tensor of a compute graph
    levels:
        The memory levels to consider. E.g., [0, 1, 2]
    """
    assert len(levels) > 0, "Can't support non-memory system."
    assert root_tensor.init is not None, "The tensor is not a result of any computation."
    graph = make_prod_consum_graph(root_tensor)
    tensors = list(reversed(graph.nodes))
    tensor_vars = [t.var for t in tensors]
    initial_bounds = []
    for t in tensor_vars:
        tensor = Tensor.tensor_cache[t]
        if tensor.init is not None:
            if len(tensor.updates):
                compute = tensor.updates[0]
            else:
                compute = tensor.init
            initial_bounds.append({l.var: l.dom for l in compute.all_loops()})
        else:
            initial_bounds.append({})
    masks = [t.init is not None for t in tensors]
    # init_trees = [MemoryLevelTree(levels, t.var) for t in tensors]
    imm_doms = graph.dominators()
    dom_map = {k.var: v.var for k, v in imm_doms.items()}
    ret = analysis.generate_merged_memory_level_trees(
        tensor_vars,
        initial_bounds,
        masks,
        dom_map,
        levels
    )
    return ret


def memory_level_tree_tiling(tree: MemoryLevelTree, tensor: Tensor, factor_dict: Dict[Loop, List[int]]):
    
    if tensor.init is None:
        raise ValueError("Can't tile for input tensors.")
    elif len(tensor.updates) == 0:
        compute = tensor.init
    elif len(tensor.updates) == 1:
        compute = tensor.updates[0]
    else:
        raise NotImplementedError("Computations with more than 1 update are not supported yet.")
    
    if tensor.init.has_reduce():
        raise RuntimeError("Don't konw how to handle reductions in init stages.")

    all_loops = compute.all_loops()
    loop_check_set = {l:None for l in all_loops}

    length = None
    for k, v in factor_dict.items():
        if length is None:
            length = len(v)
        else:
            assert length == len(v), "Tiling factors length for different loops are different."
        if k not in loop_check_set:
            raise ValueError(f"Unkonwn loop {k}")
        del loop_check_set[k]
    
    if len(loop_check_set):
        raise RuntimeError(f"Not all loops are found in tiling factors: {loop_check_set}")
    
    levels = tree.get_available_levels(tensor.var)
    assert levels == length, f"Tiling levels {levels} mismatch with factors length {length}"

    tiles = []
    for level in range(levels):
        f = []
        for l in all_loops:
            factor = factor_dict[l][level]
            new_l = Loop(factor, l.name, l.iter_type())
            f.append(new_l.iterator)
            # f.append(Iterator(Var(l.var.dtype, l.var.id), Range(0, factor, 1), l.iter_type()))
        tiles.append(f)
    loop_vars = [l.var for l in all_loops]
    tree = tree.memory_tiling(tensor.var, loop_vars, tiles)
    return tree

def infer_producer_shape(tree: MemoryLevelTree, tensor: Tensor):
    if tensor.init is None:
        raise ValueError("Can't infer producer shapes for input tensors.")
    elif len(tensor.updates) == 0:
        compute = tensor.init
    elif len(tensor.updates) == 1:
        compute = tensor.updates[0]
    else:
        raise NotImplementedError("Computations with more than 1 update are not supported yet.")

    input_tensors = compute.input_tensors()
    num_inputs = len(input_tensors)

    for idx in range(num_inputs):
        inp = input_tensors[idx]
        indices = compute.get_tensor_indices(inp)
        indices = [substitute_expr(x, tree.var_map) for x in indices]
        if inp.init is not None:
            common_mem_level = tree.least_common_ancestor(tensor.var, inp.var)
            print_ir(common_mem_level)
            if len(inp.updates):
                inp_compute = inp.updates[0]
                assert len(inp.updates) == 1
            else:
                inp_compute = inp.init
            inp_spatial_loops = inp_compute.spatial_loops()
            assert len(indices) == len(inp_spatial_loops)
            inp_spatial_loop_map = {k:v for k, v in zip(inp_spatial_loops, indices)}

            # add the new bounds of input tensor to tree
            # update_bounds = {l.var: l.dom for l in compute.all_loops()}
            # tree.set_bounds(update_bounds)
        else:
            # No need to infer bounds for input tensors
            pass
    raise NotImplementedError("The infer_bound is not implemented!")

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


def generate_tile_tensor_computation_space(tensor: Tensor, shape: List[int], levels: int):
    if tensor.init is None:
        raise ValueError("Can't tile computation for input tensors.")
    elif len(tensor.updates) == 0:
        compute = tensor.init
    elif len(tensor.updates) == 1:
        compute = tensor.updates[0]
    else:
        raise NotImplementedError("Computations with more than 1 update are not supported yet.")
    
    assert len(tensor.shape) == len(shape)
    ndim = len(shape)
    shape_dict = {}
    all_loops = compute.all_loops()
    for l, v in zip(all_loops[:ndim], shape):
        shape_dict[l] = v
    if compute.has_reduce():
        for l in compute.reduce_loops():
            extent = simplify_expr(l.dom.extent)
            assert hasattr(extent, "value"), f"Can't handle non-static bounds: {print_ir(extent, print_out=False)}"
            shape_dict[l] = extent.value
    
    tile_choices = {}
    for k, v in shape_dict.items():
        tile_choices[k] = split_to_factors(v, levels)
    
    return tile_choices
