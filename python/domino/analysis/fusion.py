from dominoc import ir, analysis
from ..program_ir import (
    ComputeLevel, MemoryLevel, Evaluate, Tensor,
    make_prod_consum_graph, print_ir)
from typing import List
import queue

__all__ = ["MemoryLevelTree", "generate_merged_memory_level_trees"]


# class MemoryLevelTree(object):
#     def __init__(self) -> None:
#         self.root = None
#         self.merged = False
#         self.initial_levels = []  # only for non-merged tree
#         self.tensor = None

#     def reset(self):
#         self.root = None
#         self.merged = False
#         self.initial_levels = []
#         self.tensor = None

#     def _copy(self):
#         new_tree = MemoryLevelTree()
#         new_tree.root = ir.replicate(self.root)
#         new_tree.merged = self.merged
#         new_tree.initial_levels = self.initial_levels
#         new_tree.tensor = self.tensor
#         return new_tree

#     def grown(self):
#         return self.root is not None and self.tensor is not None

#     def _grow(self, levels: List[int], tensor: Tensor):
#         if len(levels) == 0:
#             return None
#         leaf = tensor.var
#         cur_level = ComputeLevel(levels[0], leaf, [])
#         for level in levels:
#             cur_level = MemoryLevel(level, Evaluate(0), [cur_level])
#         return cur_level

#     def grow(self, levels: List[int], tensor: Tensor):
#         assert not self.grown(), "A grown tree can't grow anymore"
#         self.initial_levels = levels
#         self.tensor = tensor
#         self.root = self._grow(levels, tensor)

#     def cut(self, level: int):
#         new_tree = self._copy()
#         assert new_tree.root is not None and not new_tree.merged
#         while isinstance(new_tree.root, ir.MemoryLevel):
#             if new_tree.root.memory_level.value == level:
#                 new_tree.root = new_tree.root.sub_levels[0]
#                 break
#             new_tree.root = new_tree.root.sub_levels[0]
#         return new_tree

#     def _find_node(self, tensor: Tensor, level: int):
#         ans = None

#         def helper(node):
#             nonlocal ans
#             if node is None or ans is not None:
#                 return
#             has_tensor = False
#             # FIXME: ugly code to cast to Var
#             if hasattr(node.block, "get_stmt"):
#                 stmt = node.block.get_stmt()
#                 if hasattr(stmt, "expr"):
#                     expr = stmt.expr
#                     if tensor.var.same_as(expr):
#                         has_tensor = True
#             for sub in node.sub_levels:
#                 ret = helper(sub)
#                 has_tensor = has_tensor or ret
#             if has_tensor and hasattr(node, "memory_level") and node.memory_level.value == level:
#                 ans = node
#             return has_tensor
#         helper(self.root)
#         return ans

#     def merge(self, other: "MemoryLevelTree", tensor: Tensor, level: int):
#         """Self is main tree, other is secondary tree"""
#         assert not other.merged, "Can't merge a complex tree to other trees"
#         new_tree = self._copy()
#         node = new_tree._find_node(tensor, level)
#         assert node is not None, "Can't find the node to insert"
#         other = other.cut(level)
#         node.sub_levels = [other.root] + node.sub_levels
#         return new_tree

#     def _make_levels(self):
#         levels = {}
#         visit = set()
#         q = queue.Queue()
#         q.put((self.root, 0))
#         while not q.empty():
#             cur, cur_level = q.get()
#             if cur is not None:
#                 if cur_level not in levels:
#                     levels[cur_level] = []
#                 levels[cur_level].append(cur)
#                 for n in cur.sub_levels:
#                     assert n not in visit, "Duplicated level in architecture!"
#                     q.put((n, cur_level + 1))
#         return levels

#     def pretty_print(self):
#         """Function for debug print"""
#         # FIXME: this function is not written correctly...
#         indents = {}

#         def helper1(indent, cur_node):
#             if cur_node is None:
#                 return indent
#             indents[cur_node] = indent
#             child_indent = indent
#             child_size = 0
#             for sub in cur_node.sub_levels:
#                 child_size = helper1(child_indent, sub)
#                 child_indent += child_size
#             return max(child_indent - indent, 1)

#         def helper2(node):
#             if isinstance(node, ir.MemoryLevel):
#                 return f"mem@L{node.memory_level.value}"
#             elif isinstance(node, ir.ComputeLevel):
#                 line = print_ir(node.block, print_out=False)
#                 return f"comp {line[:5]}"
#             else:
#                 raise RuntimeError(f"{type(node)}, {node}")

#         helper1(0, self.root)

#         levels = self._make_levels()
#         num_levels = len(levels)
#         for level_id in range(num_levels):
#             level = levels[level_id]
#             line = ""
#             subline = ""
#             cur_indent = 0
#             for node in level:
#                 indent = indents[node]
#                 for i in range(cur_indent, indent):
#                     line += " " * 10
#                     subline += " " * 10
#                 subline += " " * 3 + "|" + " " * 6
#                 line += "{:10}".format(helper2(node))
#                 max_right = indent
#                 for child in node.sub_levels:
#                     max_right = max(max_right, indents[child])
#                 for i in range(indent, max_right):
#                     if i < max_right - 1:
#                         line += "-" * 10
#                     else:
#                         line += "-" * 5 + " " * 5
#                 cur_indent = max_right
#             print(subline)
#             print(line)

MemoryLevelTree = analysis.MemoryLevelTree


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
    masks = [t.init is not None for t in tensors]
    # init_trees = [MemoryLevelTree(levels, t.var) for t in tensors]
    imm_doms = graph.dominators()
    dom_map = {k.var: v.var for k, v in imm_doms.items()}
    ret = analysis.generate_merged_memory_level_trees(
        tensor_vars,
        masks,
        dom_map,
        levels
    )
    return ret
