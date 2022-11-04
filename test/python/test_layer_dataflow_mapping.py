import json
import os
from typing import Set, Optional, Union, List, Any
from domino import utils
from domino.graph_ir import Op, Tensor, Graph
from domino.graph_pass import set_graph_precision, GraphPrinter, GraphVisitor
from domino.utils import ONNXConvertor


def get_precision_configs(path: str):
    if not (os.path.exists(path) and os.path.isfile(path)):
        raise RuntimeError(f"The file {path} is not found")

    ret = []

    with open(path, "r") as fin:
        for line in fin:
            config = {}
            obj = json.loads(line)
            # acc = obj["acc"]
            # cost = obj["tot_cost"]
            layers = obj["layers"]
            for tid, conf in layers.items():
                if 'o_bit' in conf:
                    config[tid] = f"int{conf['o_bit']}"
                elif 'w_bit' in conf:
                    config[tid] = f"int{conf['w_bit']}"
                else:
                    raise ValueError()
            ret.append(config)

    return ret


def get_graph(path: str):
    convertor = ONNXConvertor(path, inference=True)
    graph = convertor.parse()
    return graph


conv2d_mapping_templates = {
    "NVDLA":
        lambda H, W, P, Q, K, C, R, S, stride_h, stride_w:
            ("Network sample_net {\n"
             "Layer Conv2d {\n"
             "Type: CONV\n"
             "Stride { "
             f"X: {stride_h}, Y: {stride_w} "
             "}\n"
             "Dimensions { "
             f"K: {K}, C: {C}, R: {R}, S: {S}, Y: {H}, X: {W} "
             "}\n"
             "Dataflow {\n"
             "        SpatialMap(1,1) K;\n"
             "        TemporalMap(64,64) C;\n"
             "        TemporalMap(Sz(R),Sz(R)) R;\n"
             "        TemporalMap(Sz(S),Sz(S)) S;\n"
             "        TemporalMap(Sz(R),1) Y;\n"
             "        TemporalMap(Sz(S),1) X;\n"
             "        Cluster(64, P);\n"
             "        SpatialMap(1,1) C;\n"
             "        TemporalMap(Sz(R),1) Y;\n"
             "        TemporalMap(Sz(S),1) X;\n"
             "        TemporalMap(Sz(R),Sz(R)) R;\n"
             "        TemporalMap(Sz(S),Sz(S)) S;\n"
             "}\n"
             "}\n"
             "}\n"),
    "ShiDianNao":
        lambda H, W, P, Q, K, C, R, S, stride_h, stride_w:
            ("Network sample_net {\n"
             "Layer Conv2d {\n"
             "Type: CONV\n"
             "Stride { "
             f"X: {stride_h}, Y: {stride_w} "
             "}\n"
             "Dimensions { "
             f"K: {K}, C: {C}, R: {R}, S: {S}, Y: {H}, X: {W} "
             "}\n"
             "Dataflow {\n"
             "        TemporalMap(1,1) K;\n"
             "        TemporalMap(1,1) C;\n"
             "        SpatialMap(Sz(R), 1) Y;\n"
             "        TemporalMap(8,8) X;\n"
             "        TemporalMap(Sz(R), Sz(R)) R;\n"
             "        TemporalMap(Sz(S), Sz(S)) S;\n"
             "        Cluster(8, P);\n"
             "        SpatialMap(Sz(S), 1) X;\n"
             "}\n"
             "}\n"
             "}\n"),
    "Eyeriss":
        lambda H, W, P, Q, K, C, R, S, stride_h, stride_w:
            ("Network sample_net {\n"
             "Layer Conv2d {\n"
             "Type: CONV\n"
             "Stride { "
             f"X: {stride_h}, Y: {stride_w} "
             "}\n"
             "Dimensions { "
             f"K: {K}, C: {C}, R: {R}, S: {S}, Y: {H}, X: {W} "
             "}\n"
             "Dataflow {\n"
             "        SpatialMap(1,1) Y';\n"
             "        TemporalMap(1,1) X';\n"
             "        TemporalMap(1,1) C;\n"
             "        TemporalMap(16,16) K;\n"
             "        TemporalMap(Sz(R),Sz(R)) R;\n"
             "        TemporalMap(Sz(S),Sz(S)) S;\n"
             "        Cluster(Sz(R),P);\n"
             "        SpatialMap(1,1) Y;\n"
             "        SpatialMap(1,1) R;\n"
             "        TemporalMap(Sz(S),Sz(S)) S;\n"
             "}\n"
             "}\n"
             "}\n"),
    "TPU":
        lambda H, W, P, Q, K, C, R, S, stride_h, stride_w:
            ("Network sample_net {\n"
             "Layer Conv2d {\n"
             "Type: CONV\n"
             "Stride { "
             f"X: {stride_h}, Y: {stride_w} "
             "}\n"
             "Dimensions { "
             f"K: {K}, C: {C}, R: {R}, S: {S}, Y: {H}, X: {W} "
             "}\n"
             "Dataflow {\n"
             "        TemporalMap(16,16) K;\n"
             "        SpatialMap(Sz(R),1) Y;\n"
             "        TemporalMap(Sz(S),1) X;\n"
             "        TemporalMap(1,1) C;\n"
             "        Cluster(16, P);\n"
             "        SpatialMap(1,1) K;\n"
             "        TemporalMap(Sz(R),1) Y;\n"
             "        TemporalMap(Sz(S),1) X;\n"
             "        TemporalMap(Sz(R),7) R;\n"
             "        TemporalMap(Sz(S),7) S;\n"
             "}\n"
             "}\n"
             "}\n"),
}

depthwise_conv2d_mapping_templates = {
    "ShiDianNao":
        lambda H, W, P, Q, K, M, R, S, stride_h, stride_w:
            ("Network sample_net {\n"
             "Layer DepthwiseConv2d {\n"
             "Type: DSCONV\n"
             "Stride { "
             f"X: {stride_h}, Y: {stride_w} "
             "}\n"
             "Dimensions { "
             f"K: {M}, C: {K}, R: {R}, S: {S}, Y: {H}, X: {W} "
             "}\n"
             "Dataflow {\n"
             "        TemporalMap(1,1) C;\n"
             "        SpatialMap(Sz(R), 1) Y;\n"
             "        TemporalMap(10,8) X;\n"
             "        TemporalMap(Sz(R), Sz(R)) R;\n"
             "        TemporalMap(Sz(S), Sz(S)) S;\n"
             "        Cluster(8, P);\n"
             "        SpatialMap(Sz(S), 1) X;\n"
             "}\n"
             "}\n"
             "}\n")
}


gemm_mapping_templates = {
    "Gemmini":
        lambda M, N, K:
            ("Network sample_net {\n"
             "Layer GEMM {\n"
             "Type: GEMM\n"
             "Dimensions { "
             f"K: {K}, M: {M}, N: {N} "
             "}\n"
             "Dataflow {\n"
             "        SpatialMap(32, 32) M;\n"
             "        SpatialMap(32, 32) N;\n"
             "        TemporalMap(32, 32) K;\n"
             "        TemporalMap(16, 16) M;\n"
             "        Cluster(32, P);\n"
             "        SpatialMap(16, 16) N;\n"
             "        SpatialMap(16, 16) K;\n"
             "}\n"
             "}\n"
             "}\n")
}


class LayerwiseDataflowMapping(GraphVisitor):
    def __init__(self) -> None:
        super(LayerwiseDataflowMapping, self).__init__()

    ##==------------------ General Op Visitor ------------------==##
    def visit_op(self, op: Op.NamedOp, boundary_tensors: Set[Tensor]):
        """The default visitor does noting
        """
        assert isinstance(
            op, Op.NamedOp), "Expect GraphMutator to handle NamedOp"
        if self.has_visited_op(op):
            return self.get_op_visited(op)

        if op.name == Op.OpName.ConvOp.Conv2d:
            assert len(op.inputs) >= 2
            assert len(op.outputs) >= 1
            data = op.inputs["inputs"]
            weight = op.inputs["weight"]
            output = op.outputs["output"]
            shape_dict = {}
            for k, v in data.shape_dict.items():
                shape_dict[k] = v
            for k, v in weight.shape_dict.items():
                if k in shape_dict:
                    assert shape_dict[k] == v
                else:
                    shape_dict[k] = v
            for k, v in output.shape_dict.items():
                if k == "H":
                    shape_dict["P"] = v
                elif k == "W":
                    shape_dict["Q"] = v

            H, W, P, Q, K, C, R, S = [shape_dict[x] for x in "HWPQKCRS"]
            stride_h = op.attrs["strides"].value[0].value
            stride_w = op.attrs["strides"].value[1].value

            best_runtime = float("inf")
            best_results = None
            for hw, template in conv2d_mapping_templates.items():
                mapping_contents = template(
                    H, W, P, Q, K, C, R, S, stride_h, stride_w)
                # print(mapping_contents)

                mapping_file = "sample_mapping"

                if os.path.exists(f"{mapping_file}.m") and os.path.isfile(f"{mapping_file}.m"):
                    os.remove(f"{mapping_file}.m")

                with open(f"{mapping_file}.m", "w") as fout:
                    fout.write(mapping_contents)

                maestro_path = utils.find_maestro()
                command = utils.generate_maestro_command(
                    maestro_path,
                    mapping_file,
                    1000,  # noc_bw,
                    50,  # off_chip_bw,
                    256,  # num_pes,
                    100,  # l1_size,
                    3000,  # l2_size,
                )

                results = utils.run_maestro(mapping_file, command)

                if os.path.exists(f"{mapping_file}.m") and os.path.isfile(f"{mapping_file}.m"):
                    os.remove(f"{mapping_file}.m")

                info = f"Conv({H},{W},{C},{P},{Q},{K},{R},{S},{stride_h},{stride_w})"
                tmp_res = {
                    "info": info,
                    "evaluation": results
                }
                if results.runtime[0] < best_runtime:
                    best_runtime = results.runtime[0]
                    best_results = tmp_res

                # print(info, results.runtime[0])
            res = best_results

        elif op.name == Op.OpName.ConvOp.DepthwiseConv2d:
            assert len(op.inputs) >= 2
            assert len(op.outputs) >= 1
            data = op.inputs["inputs"]
            weight = op.inputs["weight"]
            output = op.outputs["output"]
            shape_dict = {}
            for k, v in data.shape_dict.items():
                shape_dict[k] = v
            for k, v in weight.shape_dict.items():
                if k in shape_dict:
                    assert shape_dict[k] == v
                else:
                    shape_dict[k] = v
            for k, v in output.shape_dict.items():
                if k == "H":
                    shape_dict["P"] = v
                elif k == "W":
                    shape_dict["Q"] = v

            H, W, P, Q, K, M, R, S = [shape_dict[x] for x in "HWPQKMRS"]
            stride_h = op.attrs["strides"].value[0].value
            stride_w = op.attrs["strides"].value[1].value

            best_runtime = float("inf")
            best_results = None
            for hw, template in depthwise_conv2d_mapping_templates.items():
                mapping_contents = template(
                    H, W, P, Q, K, M, R, S, stride_h, stride_w)
                # print(mapping_contents)

                mapping_file = "sample_mapping"

                if os.path.exists(f"{mapping_file}.m") and os.path.isfile(f"{mapping_file}.m"):
                    os.remove(f"{mapping_file}.m")

                with open(f"{mapping_file}.m", "w") as fout:
                    fout.write(mapping_contents)

                maestro_path = utils.find_maestro()
                command = utils.generate_maestro_command(
                    maestro_path,
                    mapping_file,
                    1000,  # noc_bw,
                    50,  # off_chip_bw,
                    256,  # num_pes,
                    100,  # l1_size,
                    3000,  # l2_size,
                )

                results = utils.run_maestro(mapping_file, command)

                if os.path.exists(f"{mapping_file}.m") and os.path.isfile(f"{mapping_file}.m"):
                    os.remove(f"{mapping_file}.m")

                info = f"DepthwiseConv({H},{W},{M},{P},{Q},{K},{R},{S},{stride_h},{stride_w})"
                tmp_res = {
                    "info": info,
                    "evaluation": results
                }
                if results.runtime[0] < best_runtime:
                    best_runtime = results.runtime[0]
                    best_results = tmp_res

                # print(info, results.runtime[0])
            res = best_results
        elif op.name == Op.OpName.MatrixOp.Gemm:
            assert len(op.inputs) >= 2
            assert len(op.outputs) >= 1
            data = op.inputs["inputs"]
            weight = op.inputs["weight"]
            output = op.outputs["output"]
            shape_dict = {}

            transA = op.attrs["transA"].value.value
            transB = op.attrs["transB"].value.value

            if transA:
                K, M = data.shape
            else:
                M, K = data.shape

            if transB:
                N, K = data.shape
            else:
                K, N = data.shape

            shape_dict = {
                "M": M,
                "N": N,
                "K": K
            }

            best_runtime = float("inf")
            best_results = None
            for hw, template in gemm_mapping_templates.items():
                mapping_contents = template(M, N, K)
                # print(mapping_contents)

                mapping_file = "sample_mapping"

                if os.path.exists(f"{mapping_file}.m") and os.path.isfile(f"{mapping_file}.m"):
                    os.remove(f"{mapping_file}.m")

                with open(f"{mapping_file}.m", "w") as fout:
                    fout.write(mapping_contents)

                maestro_path = utils.find_maestro()
                command = utils.generate_maestro_command(
                    maestro_path,
                    mapping_file,
                    1000,  # noc_bw,
                    50,  # off_chip_bw,
                    256,  # num_pes,
                    100,  # l1_size,
                    3000,  # l2_size,
                )

                results = utils.run_maestro(mapping_file, command)

                if os.path.exists(f"{mapping_file}.m") and os.path.isfile(f"{mapping_file}.m"):
                    os.remove(f"{mapping_file}.m")

                info = f"GEMM({M},{N},{K})"
                tmp_res = {
                    "info": info,
                    "evaluation": results
                }
                if results.runtime[0] < best_runtime:
                    best_runtime = results.runtime[0]
                    best_results = tmp_res

                # print(info, results.runtime[0])
            res = best_results
        else:
            res = None

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
        return self.record_visited_op(op, res)

    def __call__(self, graph: Graph, specify_subgraphs: Optional[Union[Set[str], List[str]]] = None, init_state=True) -> Any:
        self.visit_graph(
            graph, specify_subgraphs=specify_subgraphs, init_state=init_state)
        return self._visited_ops


def test_layerwise_mapping():
    # config_path = "new_resnet18_pareto.json"
    # model_path = "raw_resnet18.onnx"
    # config_path = "new_mobilenetv2_pareto.json"
    # model_path = "raw_mobilenetv2.onnx"
    config_path = "new_resnet50_pareto.json"
    model_path = "raw_resnet50.onnx"

    graph = get_graph(model_path)
    configs = get_precision_configs(config_path)

    graphs = []
    for config in configs:
        new_graph = set_graph_precision(
            graph,
            config,
            graph_inputs_precision="int8",
            target_ops=[
                *Op.all_ops_in(Op.OpName.ActivationOp),
                *Op.all_ops_in(Op.OpName.PoolingOp),
                *Op.all_ops_in(Op.OpName.ScalingOp),
                *Op.all_ops_in(Op.OpName.ElementwiseOp),
                *Op.all_ops_in(Op.OpName.PadOp),
                *Op.all_ops_in(Op.OpName.ReduceOp),
                *Op.all_ops_in(Op.OpName.DimOrderOp), ]
        )
        graphs.append(new_graph)

    mapper = LayerwiseDataflowMapping()
    for graph in graphs:
        mapper(graph)
    printer = GraphPrinter()
    ret = printer(graphs[0])
    # print(ret)
    assert ret


if __name__ == "__main__":
    test_layerwise_mapping()
