from cmath import isfinite
from typing import Dict, Any, List, Set, Optional, Union

from ..base import PassBase
from ..graph_ir.op_base import OpBase
from ..graph_ir import Op, SubGraph, Graph, Tensor


class GraphVisitor(PassBase):
    def __init__(self) -> None:
        super().__init__()
        self.VTABLE = {
            Op.NamedOp: self.visit_op,
            Op.ConvOp: self.visit_conv
        }

    def get_visitor(self, op: OpBase):
        if not isinstance(op, Op.NamedOp):
            raise RuntimeError(f"Can't visit {op}")
        if type(op) not in self.VTABLE:
            return self.VTABLE[Op.NamedOp]
        else:
            return self.VTABLE[type(op)]

    def init_states(self):
        self._visited_ops = {}

    def has_visited_op(self, old_op: Op.NamedOp):
        return old_op in self._visited_ops

    def get_op_visited(self, old_op: Op.NamedOp):
        return self._visited_ops[old_op]

    def record_visited_op(self, old_op: Op.NamedOp, value: Any):
        self._visited_ops[old_op] = value
        return value

    ##==------------------ General Op Visitor ------------------==##
    def visit_op(self, op: Op.NamedOp, boundary_tensors: Set[Tensor]):
        """The default visitor does noting
        """
        assert isinstance(
            op, Op.NamedOp), "Expect GraphMutator to handle NamedOp"
        if self.has_visited_op(op):
            return self.get_op_visited(op)
        for name, input_tensor in op.inputs.items():
            if input_tensor in boundary_tensors:
                # subgraph inputs
                pass
            elif input_tensor.produce_op is not None:
                # compute op
                visitor = self.get_visitor(input_tensor.produce_op)
                visitor(input_tensor.produce_op, boundary_tensors)
            else:
                # producer op
                pass
        return self.record_visited_op(op, None)

    ##==------------------ Specific Op Visitor ------------------==##
    def visit_conv(self, op: Op.ConvOp, boundary_tensors: Set[Tensor]):
        assert isinstance(op, Op.ConvOp)
        return self.visit_op(op, boundary_tensors)

    ##==------------------ SubGraph Visitor ------------------==##

    def visit_subgraph(self, subgraph: SubGraph, init_state=False):
        if init_state:
            self.init_states()
        assert isinstance(subgraph, SubGraph)
        boundary_tensors = set()
        for name, in_tensor in subgraph.input_tensors.items():
            boundary_tensors.add(in_tensor)
        for name, out_tensor in subgraph.output_tensors.items():
            if out_tensor.produce_op is not None:
                visitor = self.get_visitor(out_tensor.produce_op)
                visitor(out_tensor.produce_op, boundary_tensors)

    ##==------------------ Graph Visitor ------------------==##

    def visit_graph(self, graph: Graph, specify_subgraphs: Optional[Union[Set[str], List[str]]] = None, init_state=True):
        if init_state:
            self.init_states()
        assert isinstance(graph, Graph)

        target_subgraphs = set(graph.subgraphs.keys(
        )) if specify_subgraphs is None else set(specify_subgraphs)
        for name, subgraph in graph.subgraphs.items():
            if name in target_subgraphs:
                self.visit_subgraph(subgraph)


class GraphMutator(PassBase):
    def __init__(self) -> None:
        super().__init__()
        self.VTABLE = {
            Op.NamedOp: self.mutate_op,
            Op.ConvOp: self.mutate_conv
        }

    def get_mutator(self, op: OpBase):
        if not isinstance(op, Op.NamedOp):
            raise RuntimeError(f"Can't mutate {op}")
        if type(op) not in self.VTABLE:
            return self.VTABLE[Op.NamedOp]
        else:
            return self.VTABLE[type(op)]

    def init_states(self):
        self._mutated_subgraphs = {}
        self._mutated_ops = {}

    def has_mutated_op(self, old_op: Op.NamedOp):
        return old_op in self._mutated_ops

    def has_mutated_subgraph(self, old_subgraph: SubGraph):
        return old_subgraph in self._mutated_subgraphs

    def get_mutated_op(self, old_op: Op.NamedOp):
        return self._mutated_ops[old_op]

    def get_mutated_subgraph(self, old_subgraph: SubGraph):
        return self._mutated_subgraphs[old_subgraph]

    def record_mutated_op(self, old_op: Op.NamedOp, new_op: Op.NamedOp):
        self._mutated_ops[old_op] = new_op
        return new_op

    def record_mutated_subgraph(self, old_subgraph: SubGraph, new_subgraph: SubGraph):
        self._mutated_subgraphs[old_subgraph] = new_subgraph
        return new_subgraph

    ##==------------------ General Op Mutator ------------------==##
    def mutate_op(self, op: Op.NamedOp, boundary_tensors: Set[Tensor]):
        assert isinstance(
            op, Op.NamedOp), "Expect GraphMutator to handle NamedOp"
        if self.has_mutated_op(op):
            return self.get_mutated_op(op)
        new_inputs = {}
        new_outputs = {}
        for name, input_tensor in op.inputs.items():
            if input_tensor in boundary_tensors:
                # subgraph boundary
                new_input_tensor = input_tensor
            elif input_tensor.produce_op is not None:
                # compute op
                mutator = self.get_mutator(input_tensor.produce_op)
                new_input_op = mutator(
                    input_tensor.produce_op, boundary_tensors)
                assert input_tensor.out_idx in new_input_op.outputs
                new_input_tensor = new_input_op.outputs[input_tensor.out_idx]
            else:
                # producer op, don't change Tensor
                new_input_tensor = input_tensor
            new_inputs[name] = new_input_tensor
        for name, output_tensor in op.outputs.items():
            new_outputs[name] = Tensor(
                output_tensor.shape,
                output_tensor.dtype,
                layout=output_tensor.layout,
                name=output_tensor.name,
                quant_params=output_tensor.quant_params,
                tensor_idx=output_tensor.tensor_idx
            )
        new_op = Op.NamedOp(
            op.name,
            new_inputs,
            new_outputs,
            op.quant_params,
            op.attrs
        )

        return self.record_mutated_op(op, new_op)

    ##==------------------ Specific Op Mutator ------------------==##
    def mutate_conv(self, op: Op.ConvOp, boundary_tensors: Set[Tensor]):
        assert isinstance(op, Op.ConvOp)
        if self.has_mutated_op(op):
            return self.get_mutated_op(op)

        new_inputs = {}
        for name, input_tensor in op.inputs.items():
            if input_tensor in boundary_tensors:
                # subgraph boundary
                new_input_tensor = input_tensor
            elif input_tensor.produce_op is not None:
                # compute op
                mutator = self.get_mutator(input_tensor.produce_op)
                new_input_op = mutator(
                    input_tensor.produce_op, boundary_tensors)
                assert input_tensor.out_idx in new_input_op.outputs
                new_input_tensor = new_input_op.outputs[input_tensor.out_idx]
            else:
                # producer op, don't change Tensor
                new_input_tensor = input_tensor
            new_inputs[name] = new_input_tensor

        new_op = Op.ConvOp(
            new_inputs["inputs"],
            new_inputs["weight"],
            bias=new_inputs.get("bias", None),
            strides=op.strides,
            padding=op.padding,
            dilation=op.dilation,
            output_dtype=op.outputs["output"].dtype,
            output_layout=op.outputs["output"].layout,
            output_quant_params=op.outputs["output"].quant_params,
            conv_quant_params=op.quant_params,
            attrs=op.attrs
        )

        return self.record_mutated_op(op, new_op)

    def mutate_subgraph(self, subgraph: SubGraph, init_state=False):
        if init_state:
            self.init_states()
        assert isinstance(subgraph, SubGraph)
        boundary_tensors = set()
        for name, in_tensor in subgraph.input_tensors.items():
            boundary_tensors.add(in_tensor)
        new_output_tensors = {}
        for name, out_tensor in subgraph.output_tensors.items():
            if out_tensor.produce_op is not None:
                mutator = self.get_mutator(out_tensor.produce_op)
                new_op = mutator(out_tensor.produce_op, boundary_tensors)
                assert out_tensor.out_idx in new_op.outputs
                new_output_tensors[name] = new_op.outputs[out_tensor.out_idx]
        return SubGraph(
            subgraph.input_tensors,
            new_output_tensors
        )

    def mutate_graph(self, graph: Graph, specify_subgraphs: Optional[Union[Set[str], List[str]]] = None, init_state=True):
        if init_state:
            self.init_states()
        assert isinstance(graph, Graph)
        target_subgraphs = set(graph.subgraphs.keys(
        )) if specify_subgraphs is None else set(specify_subgraphs)
        new_subgraphs = {}
        new_inputs = {}
        new_outputs = {}
        mutate_tensors = {}
        for name, subgraph in graph.subgraphs.items():
            if name in target_subgraphs:
                new_subgraph = self.mutate_subgraph(subgraph)
                for tname, old_tensor in subgraph.output_tensors.items():
                    assert tname in new_subgraph.output_tensors
                    mutate_tensors[old_tensor] = new_subgraph.output_tensors[tname]
            else:
                new_subgraph = SubGraph(
                    subgraph.input_tensors, subgraph.output_tensors)
            new_subgraphs[name] = new_subgraph

        for name, subgraph in graph.subgraphs.items():
            new_input_tensors = {}
            for tname, old_tensor in subgraph.input_tensors.items():
                if old_tensor in mutate_tensors:
                    new_input_tensors[tname] = mutate_tensors[old_tensor]
                else:
                    new_input_tensors[tname] = old_tensor
            subgraph.input_tensors = new_input_tensors

        for name, tensor in graph.graph_inputs.items():
            if tensor in mutate_tensors:
                new_inputs[name] = mutate_tensors[tensor]
            else:
                new_inputs[name] = tensor

        for name, tensor in graph.graph_outputs.items():
            if tensor in mutate_tensors:
                new_outputs[name] = mutate_tensors[tensor]
            else:
                new_outputs[name] = tensor

        return Graph(new_subgraphs, new_inputs, new_outputs)
