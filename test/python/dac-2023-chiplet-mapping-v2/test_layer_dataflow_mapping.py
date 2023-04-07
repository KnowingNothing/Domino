import json
import os
import networkx as nx
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
             "        SpatialMap(1,1) K;\n"
             "        TemporalMap(128,128) C;\n"
             "        TemporalMap(Sz(R),Sz(R)) R;\n"
             "        TemporalMap(Sz(S),Sz(S)) S;\n"
             "        TemporalMap(Sz(R),1) Y;\n"
             "        TemporalMap(Sz(S),1) X;\n"
             "        Cluster(128, P);\n"
             "        SpatialMap(1,1) C;\n"
             "        TemporalMap(Sz(R),1) Y;\n"
             "        TemporalMap(Sz(S),1) X;\n"
             "        TemporalMap(Sz(R),Sz(R)) R;\n"
             "        TemporalMap(Sz(S),Sz(S)) S;\n"
             "}\n"
             "}\n"
             "}\n"),
    "MaestroNVDLA":
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
             "        TemporalMap(128,128) X;\n"
             "        TemporalMap(Sz(R), Sz(R)) R;\n"
             "        TemporalMap(Sz(S), Sz(S)) S;\n"
             "        Cluster(128, P);\n"
             "        SpatialMap(Sz(S), 1) X;\n"
             "}\n"
             "}\n"
             "}\n"),
    # "Eyeriss":
    #     lambda H, W, P, Q, K, C, R, S, stride_h, stride_w:
    #         ("Network sample_net {\n"
    #          "Layer Conv2d {\n"
    #          "Type: CONV\n"
    #          "Stride { "
    #          f"X: {stride_h}, Y: {stride_w} "
    #          "}\n"
    #          "Dimensions { "
    #          f"K: {K}, C: {C}, R: {R}, S: {S}, Y: {H}, X: {W} "
    #          "}\n"
    #          "Dataflow {\n"
    #          "        SpatialMap(1,1) Y';\n"
    #          "        TemporalMap(1,1) X';\n"
    #          "        TemporalMap(1,1) C;\n"
    #          "        TemporalMap(16,16) K;\n"
    #          "        TemporalMap(Sz(R),Sz(R)) R;\n"
    #          "        TemporalMap(Sz(S),Sz(S)) S;\n"
    #          "        Cluster(Sz(R),P);\n"
    #          "        SpatialMap(1,1) Y;\n"
    #          "        SpatialMap(1,1) R;\n"
    #          "        TemporalMap(Sz(S),Sz(S)) S;\n"
    #          "}\n"
    #          "}\n"
    #          "}\n"),
    # "TPU":
    #     lambda H, W, P, Q, K, C, R, S, stride_h, stride_w:
    #         ("Network sample_net {\n"
    #          "Layer Conv2d {\n"
    #          "Type: CONV\n"
    #          "Stride { "
    #          f"X: {stride_h}, Y: {stride_w} "
    #          "}\n"
    #          "Dimensions { "
    #          f"K: {K}, C: {C}, R: {R}, S: {S}, Y: {H}, X: {W} "
    #          "}\n"
    #          "Dataflow {\n"
    #          "        TemporalMap(16,16) K;\n"
    #          "        SpatialMap(Sz(R),1) Y;\n"
    #          "        TemporalMap(Sz(S),1) X;\n"
    #          "        TemporalMap(1,1) C;\n"
    #          "        Cluster(16, P);\n"
    #          "        SpatialMap(1,1) K;\n"
    #          "        TemporalMap(Sz(R),1) Y;\n"
    #          "        TemporalMap(Sz(S),1) X;\n"
    #          "        TemporalMap(Sz(R),7) R;\n"
    #          "        TemporalMap(Sz(S),7) S;\n"
    #          "}\n"
    #          "}\n"
    #          "}\n"),
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
             "        TemporalMap(128,128) X;\n"
             "        TemporalMap(Sz(R), Sz(R)) R;\n"
             "        TemporalMap(Sz(S), Sz(S)) S;\n"
             "        Cluster(128, P);\n"
             "        SpatialMap(Sz(S), 1) X;\n"
             "}\n"
             "}\n"
             "}\n")
}


gemm_mapping_templates = {
    "TPU":
        lambda M, N, K:
            ("Network sample_net {\n"
             "Layer GEMM {\n"
             "Type: GEMM\n"
             "Dimensions { "
             f"K: {K}, M: {M}, N: {N} "
             "}\n"
             "Dataflow {\n"
             "        SpatialMap(1, 1) M;\n"
             "        TemporalMap(128, 128) N;\n"
             "        TemporalMap(16, 16) K;\n"
             "        Cluster(128, P);\n"
             "        SpatialMap(1, 1) N;\n"
             "        TemporalMap(1, 1) M;\n"
             "        TemporalMap(16, 16) K;\n"
             "}\n"
             "}\n"
             "}\n")
}


MAX_BW = 81920000


def compute_area_external(num_pe, l1_size, l2_size, precision):
    """return um^2"""
    alpha = 2.3141918
    TABLE = {
        "444": 282/alpha/alpha/alpha,
        "448": 282/alpha/alpha,
        "484": 282/alpha/alpha,
        "488": 282/alpha,
        "884": 282/alpha,
        "888": 282
    }
    # MAC_AREA_INT8=282
    # MAC_AREA_INT32=3495
    MAC_AREA = TABLE[precision]
    BUF_AREA_perbit = 0.086
    buf_size = l1_size * num_pe + l2_size
    area = num_pe * MAC_AREA + buf_size * BUF_AREA_perbit * 8
    return area


class LayerwiseDataflowMapping(GraphVisitor):
    def __init__(self) -> None:
        super(LayerwiseDataflowMapping, self).__init__()
        self.noc_bw = MAX_BW  # noc_bw,
        self.off_chip_bw = MAX_BW  # off_chip_bw,
        self.num_pes = 128*128  # num_pes,
        self.l1_size = 4000000  # l1_size,
        self.l2_size = 24000000  # l2_size,

    def _clear_states(self):
        self._visited_conv2d_shape = {}
        self._visited_depthwise_shape = {}
        self._graph = nx.Graph()
        self._unique_node_id = 0
        self._transit_node_mappings = {}

    def _make_graph_node(self, info, input_elements, input_dtype, weight_dtype, output_dtype, hw_name, configs):
        input_width = int(str(input_dtype).replace('int', ''))
        weight_width = int(str(input_dtype).replace('int', ''))
        output_width = int(str(output_dtype).replace('int', ''))
        precision = f"{min(input_width, weight_width)}{max(input_width, weight_width)}{output_width}"
        area = compute_area_external(
            self.num_pes,
            configs.l1_size[0],
            configs.l2_size[0],
            precision
        )

        hw_configs = {
            "name": hw_name,
            "runtime": configs.runtime[0],
            # "area": area,
            # "L1": configs.l1_size[0],
            # "L2": configs.l2_size[0],
            "memory_usage": configs.l1_size[0] * self.num_pes + configs.l2_size[0]
            # "power": configs.power[0],
            # "input_dtype": str(input_dtype),
            # "weight_dtype": str(weight_dtype),
            # "output_dtype": str(output_dtype)
        }

        self._unique_node_id += 1
        return (self._unique_node_id, {"layer_info": info, "input_elements": input_elements, "hw_configs": hw_configs})

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

            shape_key = (H, W, P, Q, K, C, R, S, stride_h, stride_w)
            if shape_key in self._visited_conv2d_shape:
                res = self._visited_conv2d_shape[shape_key]
            else:
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
                        self.noc_bw,
                        self.off_chip_bw,  # off_chip_bw,
                        self.num_pes,  # num_pes,
                        self.l1_size,  # l1_size,
                        self.l2_size,  # l2_size,
                    )

                    results = utils.run_maestro(mapping_file, command)

                    if os.path.exists(f"{mapping_file}.m") and os.path.isfile(f"{mapping_file}.m"):
                        os.remove(f"{mapping_file}.m")

                    info = f"Conv({H},{W},{C},{P},{Q},{K},{R},{S},{stride_h},{stride_w})"
                    tmp_res = {
                        "info": info,
                        "hw_name": hw,
                        "evaluation": results
                    }
                    if results.runtime[0] < best_runtime:
                        best_runtime = results.runtime[0]
                        best_results = tmp_res

                    # print(info, results.runtime[0])
                assert best_results is not None
                res = best_results
                node = self._make_graph_node(
                    res["info"],
                    H * W * C,
                    data.dtype,
                    weight.dtype,
                    output.dtype,
                    res["hw_name"],
                    res["evaluation"]
                )
                self._graph.add_nodes_from([node])

                for name, input_tensor in op.inputs.items():
                    if input_tensor in boundary_tensors:
                        # subgraph inputs
                        pass
                    elif input_tensor.produce_op is not None:
                        # compute op
                        assert input_tensor.produce_op in self._transit_node_mappings

                        for parent_node in self._transit_node_mappings[input_tensor.produce_op]:
                            parent_node_name, parent_node_contents = parent_node
                            # parent_node_contents["output_elements"] / res["evaluation"].runtime[0]
                            edge_weight = -1
                            self._graph.add_edge(
                                parent_node_name, node[0], weight=edge_weight)
                    else:
                        # producer op
                        pass

                self._transit_node_mappings[op] = [node]

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

            shape_key = (H, W, P, Q, K, M, R, S, stride_h, stride_w)
            if shape_key in self._visited_depthwise_shape:
                res = self._visited_depthwise_shape[shape_key]
            else:
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
                        self.noc_bw,
                        self.off_chip_bw,  # off_chip_bw,
                        self.num_pes,  # num_pes,
                        self.l1_size,  # l1_size,
                        self.l2_size,  # l2_size,
                    )

                    results = utils.run_maestro(mapping_file, command)

                    if os.path.exists(f"{mapping_file}.m") and os.path.isfile(f"{mapping_file}.m"):
                        os.remove(f"{mapping_file}.m")

                    info = f"DepthwiseConv({H},{W},{M},{P},{Q},{K},{R},{S},{stride_h},{stride_w})"
                    tmp_res = {
                        "info": info,
                        "hw_name": hw,
                        "evaluation": results
                    }
                    if results.runtime[0] < best_runtime:
                        best_runtime = results.runtime[0]
                        best_results = tmp_res

                    # print(info, results.runtime[0])
                assert best_results is not None
                res = best_results
                node = self._make_graph_node(
                    res["info"],
                    H * W * K // M,
                    data.dtype,
                    weight.dtype,
                    output.dtype,
                    res["hw_name"],
                    res["evaluation"]
                )
                self._graph.add_nodes_from([node])

                for name, input_tensor in op.inputs.items():
                    if input_tensor in boundary_tensors:
                        # subgraph inputs
                        pass
                    elif input_tensor.produce_op is not None:
                        # compute op
                        assert input_tensor.produce_op in self._transit_node_mappings

                        for parent_node in self._transit_node_mappings[input_tensor.produce_op]:
                            parent_node_name, parent_node_contents = parent_node
                            # parent_node_contents["output_elements"] / res["evaluation"].runtime[0]
                            edge_weight = -1
                            self._graph.add_edge(
                                parent_node_name, node[0], weight=edge_weight)
                    else:
                        # producer op
                        pass

                self._transit_node_mappings[op] = [node]
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
                    self.noc_bw,
                    self.off_chip_bw,  # off_chip_bw,
                    self.num_pes,  # num_pes,
                    self.l1_size,  # l1_size,
                    self.l2_size,  # l2_size,
                )

                results = utils.run_maestro(mapping_file, command)

                if os.path.exists(f"{mapping_file}.m") and os.path.isfile(f"{mapping_file}.m"):
                    os.remove(f"{mapping_file}.m")

                info = f"GEMM({M},{N},{K})"
                tmp_res = {
                    "info": info,
                    "hw_name": hw,
                    "evaluation": results
                }
                if results.runtime[0] < best_runtime:
                    best_runtime = results.runtime[0]
                    best_results = tmp_res

                # print(info, results.runtime[0])
            assert best_results is not None
            res = best_results
            node = self._make_graph_node(
                res["info"],
                M * K,
                data.dtype,
                weight.dtype,
                output.dtype,
                res["hw_name"],
                res["evaluation"]
            )
            self._graph.add_nodes_from([node])

            for name, input_tensor in op.inputs.items():
                if input_tensor in boundary_tensors:
                    # subgraph inputs
                    pass
                elif input_tensor.produce_op is not None:
                    # compute op
                    assert input_tensor.produce_op in self._transit_node_mappings

                    for parent_node in self._transit_node_mappings[input_tensor.produce_op]:
                        parent_node_name, parent_node_contents = parent_node
                        # parent_node_contents["output_elements"] / res["evaluation"].runtime[0]
                        edge_weight = -1
                        self._graph.add_edge(
                            parent_node_name, node[0], weight=edge_weight)
                else:
                    # producer op
                    pass

            self._transit_node_mappings[op] = [node]
        else:
            # omit this op in mapping
            parent_nodes = []
            for name, input_tensor in op.inputs.items():
                if input_tensor in boundary_tensors:
                    # subgraph inputs
                    pass
                elif input_tensor.produce_op is not None:
                    # compute op
                    assert input_tensor.produce_op in self._transit_node_mappings

                    parent_nodes.extend(
                        self._transit_node_mappings[input_tensor.produce_op])
                else:
                    # producer op
                    pass

            self._transit_node_mappings[op] = parent_nodes
            res = None

        return self.record_visited_op(op, res)

    def __call__(self, graph: Graph, specify_subgraphs: Optional[Union[Set[str], List[str]]] = None, init_state=True) -> Any:
        self._clear_states()
        self.visit_graph(
            graph, specify_subgraphs=specify_subgraphs, init_state=init_state)
        return self._graph


def post_process(nx_graph):
    REAL_BW = 1.5e12  # byte/s
    REAL_FREQ = 200e6  # Hz
    min_runtime = float("inf")
    min_layer = None
    min_hw = None
    max_runtime = 0
    max_layer = None
    max_hw = None

    max_overhead = 0
    # max_input_elements = 0
    # max_input_dtype = 0

    # min_area = float("inf")
    # max_area = 0
    max_memory_usage = {
        "TPU": 0,
        "MaestroNVDLA": 0,
        "ShiDianNao": 0
    }
    for node in nx_graph.nodes:
        # print("l1_size:", nx_graph.nodes[node]["hw_configs"]["L1"])
        # print("l2_size:", nx_graph.nodes[node]["hw_configs"]["L2"])
        print("memory usage:", nx_graph.nodes[node]["hw_configs"]["memory_usage"])
        hw = nx_graph.nodes[node]["hw_configs"]["name"]
        # area = nx_graph.nodes[node]["hw_configs"]["area"]
        layer_info = nx_graph.nodes[node]["layer_info"]
        runtime = nx_graph.nodes[node]["hw_configs"]["runtime"]
        input_elements = nx_graph.nodes[node]["input_elements"]
        # input_dtype = int(
        #     nx_graph.nodes[node]["hw_configs"]["input_dtype"].replace("int", "")) / 8
        input_dtype = 1 # byte

        if runtime < min_runtime:
            min_runtime = runtime
            min_layer = layer_info
            min_hw = hw

        # if area < min_area:
        #     min_area = area

        # if area > max_area:
        #     max_area = area

        max_memory_usage[hw] = max(max_memory_usage[hw], nx_graph.nodes[node]["hw_configs"]["memory_usage"])

        if runtime > max_runtime:
            max_runtime = runtime
            max_layer = layer_info
            max_hw = hw

        overhead = runtime / REAL_FREQ + \
            (input_elements * input_dtype) / REAL_BW
        if overhead > max_overhead:
            max_overhead = overhead
            # max_input_elements = nx_graph.nodes[node]["input_elements"]
            # max_input_dtype = int(nx_graph.nodes[node]["hw_configs"]["input_dtype"].replace("int", "")) / 8

    print(min_runtime, max_runtime, max_runtime / min_runtime)
    print("min runtime:", min_layer, min_hw)
    print("max runtime:", max_layer, max_hw)

    # print("min area:", min_area/1e6)
    # print("max area:", max_area/1e6)
    print("max memory usage:")
    print(max_memory_usage)

    # align memory usage
    for node in nx_graph.nodes:
        hw = nx_graph.nodes[node]["hw_configs"]["name"]
        nx_graph.nodes[node]["hw_configs"]["memory_usage"] = max_memory_usage[hw]

    target_overhead = max_overhead

    for edge in nx_graph.edges:
        if edge[0] > edge[1]:
            continue
        node = edge[1]
        runtime = nx_graph.nodes[node]["hw_configs"]["runtime"]
        compute_overhead = runtime / REAL_FREQ
        remain_overhead = target_overhead - compute_overhead
        input_elements = nx_graph.nodes[node]["input_elements"]
        # input_dtype = int(
        #     nx_graph.nodes[node]["hw_configs"]["input_dtype"].replace("int", "")) / 8
        input_dtype = 1 # byte
        input_bytes = input_elements * input_dtype
        required_bw = input_bytes / remain_overhead
        nx_graph[edge[0]][node]["weight"] = required_bw

    # for edge in nx_graph.edges:
    #     if edge[0] > edge[1]:
    #         continue
    #     print(edge, nx_graph[edge[0]][edge[1]])


def test_layerwise_mapping():
    paths = [
        ("new_resnet18_pareto.json", "raw_resnet18.onnx"),
        ("new_mobilenetv2_pareto.json", "raw_mobilenetv2.onnx"),
        ("new_resnet50_pareto.json", "raw_resnet50.onnx"),
        ("w_a_joint_yolov5_pareto.json", "yolov5s_640x640.simplify.onnx")
    ]
    for config_path, model_path in paths:
    # config_path = "new_resnet18_pareto.json"
    # model_path = "raw_resnet18.onnx"
    # config_path = "new_mobilenetv2_pareto.json"
    # model_path = "raw_mobilenetv2.onnx"
    # config_path = "new_resnet50_pareto.json"
    # model_path = "raw_resnet50.onnx"
    # config_path = "w_a_joint_yolov5_pareto.json"
    # model_path = "yolov5s_640x640.simplify.onnx"

        graph = get_graph(f"../{model_path}")
        configs = get_precision_configs(f"../{config_path}")

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
        for i, graph in enumerate(graphs):
            nx_graph = mapper(graph)
            post_process(nx_graph)
            nx.write_gpickle(
                nx_graph, f"{config_path.replace('.json', '')}_{i}.pkl")
            break
        # printer = GraphPrinter()
        # ret = printer(graphs[0])
        # print(ret)
        # assert ret


if __name__ == "__main__":
    test_layerwise_mapping()
