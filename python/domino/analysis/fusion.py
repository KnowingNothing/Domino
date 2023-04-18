from dominoc import ir, analysis
from ..program_ir.arch import (
    ComputeLevel, MemoryLevel)
from ..program_ir.block import AtomBlock
from ..program_ir.stmt import (Evaluate, Var, Iterator, Range,
                               NdStore)
from ..program_ir.simplify import (simplify_expr, substitute_expr)
from ..program_ir.dsl import (Tensor, Loop, TreeContainer, SyntaxContext, ScopeBlockContext,
                              TileBlockContext, StmtBlockContext, make_prod_consum_graph)
from ..program_ir.functional import print_ir
from ..passes import ProdConsumGraph
from typing import List, Dict
import queue
import math
from functools import lru_cache

__all__ = [
    # Memory Level Tree API
    "MemoryLevelTree",
    "create_tree",
    "generate_merged_memory_level_trees",
    "memory_level_tree_tiling",
    "infer_producer_shape",
    # TreeContainer API
    "merge_tree",
    "generate_fusion_plans",
]

## ==---------------------------------------------==##
##             Memory Level Tree API               ##
## ==---------------------------------------------==##

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
        raise NotImplementedError(
            "Computations with more than 1 update are not supported yet.")

    if tensor.init.has_reduce():
        raise RuntimeError(
            "Don't konw how to handle reductions in init stages.")
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
        raise NotImplementedError(
            "Computations with more than 1 update are not supported yet.")

    if tensor.init.has_reduce():
        raise RuntimeError(
            "Don't konw how to handle reductions in init stages.")

    all_loops = compute.all_loops()
    loop_check_set = {l: None for l in all_loops}

    length = None
    for k, v in factor_dict.items():
        if length is None:
            length = len(v)
        else:
            assert length == len(
                v), "Tiling factors length for different loops are different."
        if k not in loop_check_set:
            raise ValueError(f"Unkonwn loop {k}")
        del loop_check_set[k]

    if len(loop_check_set):
        raise RuntimeError(
            f"Not all loops are found in tiling factors: {loop_check_set}")

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
        raise NotImplementedError(
            "Computations with more than 1 update are not supported yet.")

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
            inp_spatial_loop_map = {k: v for k,
                                    v in zip(inp_spatial_loops, indices)}

            # add the new bounds of input tensor to tree
            # update_bounds = {l.var: l.dom for l in compute.all_loops()}
            # tree.set_bounds(update_bounds)
        else:
            # No need to infer bounds for input tensors
            pass
    raise NotImplementedError("The infer_bound is not implemented!")


## ==---------------------------------------------==##
##               TreeContainer API                 ##
## ==---------------------------------------------==##

def check_simple_tree(tree: TreeContainer):
    assert isinstance(tree, TreeContainer)
    level = []

    def walker(cur):
        if len(cur.children) == 0:
            level.append(cur)
            return
        if len(cur.children) > 1:
            for c in cur.children:
                assert not isinstance(
                    c.ctx, TileBlockContext), "Not a simple tree"
                assert len(c.children) == 0
        assert isinstance(cur.ctx, TileBlockContext), "Not a simple tree"
        level.append(cur)

    tree.walk(walker)
    return level


def check_position_in_tree(tree: TreeContainer, position: TreeContainer):
    assert isinstance(tree, TreeContainer)
    assert isinstance(position, TreeContainer)
    found = False

    def walker(cur):
        nonlocal found
        if cur == position:
            found = True
    tree.walk(walker)
    assert found, f"The given position {position} is not part of tree {tree}"


def find_path(tree: TreeContainer, position: TreeContainer):
    assert isinstance(tree, TreeContainer)
    assert isinstance(position, TreeContainer)
    path = []
    found = set()

    def walker(cur):
        nonlocal path
        nonlocal found
        if cur == position:
            found.add(cur)
            path.append(cur)
        for c in cur.children:
            if c in found:
                found.add(cur)
                path.append(cur)
    tree.post_order_walk(walker)
    return list(reversed(path))


def tensor_in_tree(tree: TreeContainer, tensor: Tensor):
    """Find the tensor of the same name in the tree.
        Note that we only use name for comparison

    Args:
        tree (TreeContainer): the tree
        tensor (Tensor): the tensor
    """
    assert isinstance(tree, TreeContainer)
    assert isinstance(tensor, Tensor)

    found = False

    def walker(cur):
        nonlocal found
        if isinstance(cur.ctx, StmtBlockContext) and isinstance(cur.ctx.stmt, ir.NdStore):
            # TODO: use var for comparison
            # if cur.ctx.stmt.mem_ref.var.same_as(tensor.var):
            #     found = True
            if cur.ctx.stmt.mem_ref.var.id.value == tensor.var.id.value:
                found = True

    tree.walk(walker)
    return found


def find_position(tree: TreeContainer, level: int, tensor: Tensor):
    assert isinstance(tree, TreeContainer)
    assert isinstance(level, int)
    assert isinstance(tensor, Tensor)
    assert (tensor_in_tree(tree, tensor))

    position = None

    def helper(cur, dep):
        nonlocal position
        if dep == level:
            if tensor_in_tree(cur, tensor):
                position = cur
            return
        else:
            for c in cur.children:
                if isinstance(c.ctx, ScopeBlockContext):
                    helper(c, dep)
                else:
                    helper(c, dep + 1)

    helper(tree, 0)
    assert position is not None
    return position


def merge_tree(root: TreeContainer, main_tree: TreeContainer, branch_tree: TreeContainer, position: TreeContainer, scope: str):
    """Merge Tree Function for TreeContainer

    Args:
        main_tree (TreeContainer): the tree to merge to
        branch_tree (TreeContainer): the tree to be merged
        position (TreeContainer): the merge position in main_tree
    """
    assert main_tree in root.children
    assert branch_tree in root.children
    assert isinstance(main_tree, TreeContainer)
    assert isinstance(branch_tree, TreeContainer)
    branch_levels = check_simple_tree(branch_tree)
    check_position_in_tree(main_tree, position)
    path = find_path(main_tree, position)
    level = 0
    non_scope_path = []
    for p in path:
        if not isinstance(p.ctx, ScopeBlockContext):
            level += 1
            non_scope_path.append(p)

    assert len(branch_levels) > level, "The given position is out of range"
    scope_ctx = ScopeBlockContext(position.ctx.ir_builder, scope)
    scope_node = TreeContainer(scope_ctx)
    scope_node.children = position.children
    position.children = [scope_node]
    scope_node.add_front_child(branch_levels[level])
    for l in range(level):
        non_scope_path[l].ctx.merge_loops(branch_levels[l].ctx)
    # delete the original branch_tree
    new_children = []
    for c in root.children:
        if c != branch_tree:
            new_children.append(c)
    root.children = new_children


class FusionPoint:
    def __init__(self, tensor: Tensor, level: int, scope: str):
        self.tensor = tensor
        self.level = level
        self.scope = scope

    def __str__(self):
        obj = {
            "tensor": self.tensor.name,
            "level": self.level,
            "scope": self.scope
        }
        return str(obj)

    def __repr__(self):
        return str(self)


class FusionPlan:
    def __init__(self, mapping=None, tensor_order=None):
        if mapping is None:
            self.mapping = {}
        else:
            self.mapping = mapping
        if tensor_order is None:
            self.tensor_order = []
        else:
            self.tensor_order = tensor_order

    def __setitem__(self, key: Tensor, value: FusionPoint):
        assert isinstance(key, Tensor)
        assert isinstance(value, FusionPoint)
        if key not in self.mapping:
            # only record first come
            self.tensor_order.append(key)
        self.mapping[key] = value

    def __getitem__(self, key: Tensor):
        assert isinstance(key, Tensor)
        return self.mapping[key]

    def __contains__(self, key: Tensor):
        return key in self.mapping

    def clone(self):
        return FusionPlan({k: v for k, v in self.mapping.items()}, [x for x in self.tensor_order])

    def __str__(self):
        return str({t.name: v for t,v in self.mapping.items()})

    def __repr__(self):
        return str(self)

    def apply(self, output_tensors, ctx):
        if isinstance(output_tensors, Tensor):
            output_tensors = [output_tensors]
        assert isinstance(output_tensors, (list, tuple))
        assert len(output_tensors) == 1, "Only support one output Tensor"
        graph = make_prod_consum_graph(output_tensors[0])
        tensors = []
        for t in graph.nodes:
            if not graph.is_input_tensor(t):
                tensors.append(t)
        assert len(tensors) == len(ctx.stack[0].children), f"{len(tensors)} vs. {len(ctx.stack[0].children)}"

        tensor2loop = {k.var.id.value: v for k,
                       v in zip(tensors, ctx.stack[0].children)}
        for t in self.tensor_order:
            fuse_point = self.mapping[t]
            if fuse_point.level == -1:
                # nothing to do
                continue
            main_tree = tensor2loop[fuse_point.tensor.var.id.value]
            branch_tree = tensor2loop[t.var.id.value]
            position = find_position(
                main_tree, fuse_point.level, fuse_point.tensor)
            merge_tree(ctx.stack[0], main_tree,
                       branch_tree, position, fuse_point.scope)
            tensor2loop[t.var.id.value] = main_tree


class FusionState:
    def __init__(self, main_tensor: Tensor, tensor2loop: Dict[Tensor, TreeContainer], ctx):
        self.main_tensor = main_tensor
        self.tensor2loop = tensor2loop
        self.ctx = ctx


def generate_fusion_plans(output_tensors, total_levels: int):
    if isinstance(output_tensors, Tensor):
        output_tensors = [output_tensors]
    assert isinstance(output_tensors, (list, tuple))
    assert len(output_tensors) == 1, "Only support one output Tensor"
    graph = make_prod_consum_graph(output_tensors[0])
    tensors = []
    for t in graph.nodes:
        if not graph.is_input_tensor(t):
            tensors.append(t)
    dominators = graph.dominators()

    tensor2idx = {t: i for i, t in enumerate(graph.nodes)}
    full_dominate = {
        i: {j: dominators[i] == j for j in graph.nodes} for i in graph.nodes}
    # i is dominated by j
    for k in graph.nodes:
        for i in graph.nodes:
            for j in graph.nodes:
                full_dominate[i][j] = full_dominate[i][j] or (
                    full_dominate[i][k] and full_dominate[k][j])

    def helper(t: Tensor, cur_plan: FusionPlan, visit, ans):
        if len(visit) == len(tensors):
            ans.append(cur_plan)
            return
        if t in visit:
            return
        next_visit = set(visit)
        next_visit.add(t)

        assert t in dominators
        direct_dom = dominators[t]
        direct_dom_position = tensor2idx[direct_dom]
        non_dom_consum_max_position = direct_dom_position
        consum_max_position = direct_dom_position
        if t in graph.feed_links:
            for cons in graph.feed_links[t]:
                if cons in cur_plan:
                    if not full_dominate[t][cons]:
                        non_dom_consum_max_position = max(
                            non_dom_consum_max_position, tensor2idx[cur_plan[cons].tensor])
                    consum_max_position = max(
                        consum_max_position, tensor2idx[cur_plan[cons].tensor])

        if non_dom_consum_max_position <= direct_dom_position:
            # start from direct dom
            start_idx = direct_dom_position
            start_tensor = direct_dom
        else:
            start_idx = consum_max_position
            start_tensor = graph.nodes[start_idx]

        level_deeper = total_levels
        level_topper = 0
        if start_tensor in cur_plan and start_tensor != cur_plan[start_tensor].tensor:
            level_topper = cur_plan[start_tensor].level + 1

        if t in graph.feed_links:
            for cons in graph.feed_links[t]:
                if cons in cur_plan:
                    if cur_plan[cons].tensor == start_tensor and start_tensor in cur_plan and start_tensor != cur_plan[start_tensor].tensor:
                        level_deeper = min(level_deeper, cur_plan[cons].level)

        assert full_dominate[t][start_tensor], "Choose a tensor but not a dominator"
        if start_tensor == t:
            # the last tensor
            if start_tensor in cur_plan:
                start_tensor = cur_plan[start_tensor].tensor
                level_deeper = total_levels
                level_topper = cur_plan[start_tensor].level + 1
            else:
                next_plan = cur_plan.clone()
                next_plan[t] = FusionPoint(start_tensor, -1, "Sequential")
                if t in graph.read_links:
                    for inp in graph.read_links[t]:
                        if not graph.is_input_tensor(inp):
                            visit.add(t)
                            helper(inp, next_plan, next_visit, ans)

        else:
            while True:
                # not the last tensor
                # each corresponds to one fusion possibility
                for l in range(level_topper, min(level_deeper + 1, total_levels)):
                    for scope in ["Sequential", "Sharing", "Pipeline", "Parallel"]:
                        next_plan = cur_plan.clone()
                        next_plan[t] = FusionPoint(start_tensor, l, scope)
                        if t in graph.read_links:
                            for inp in graph.read_links[t]:
                                if not graph.is_input_tensor(inp):
                                    helper(inp, next_plan, next_visit, ans)
                if start_tensor in cur_plan and start_tensor != cur_plan[start_tensor].tensor:
                    start_tensor = cur_plan[start_tensor].tensor
                    level_deeper = level_topper - 1
                    level_topper = cur_plan[start_tensor].level + 1
                else:
                    break

    empty_plan = FusionPlan()
    visit = set()
    all_plans = []
    helper(output_tensors[0], empty_plan, visit, all_plans)

    return all_plans
