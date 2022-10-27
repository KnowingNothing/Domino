import domino
from domino.graph_pass import GraphMutator, GraphVisitor, GraphPrinter
from domino.utils import ONNXConvertor


def get_graph(path: str):
    convertor = ONNXConvertor(path, inference=True)
    graph = convertor.parse()
    return graph


def test_basic_usage():
    model_path = "raw_resnet18.onnx"
    
    graph = get_graph(model_path)
    visitor = GraphVisitor()
    mutator = GraphMutator()
    
    visitor.visit_graph(graph)
    new_graph = mutator.mutate_graph(graph)
    
    assert new_graph
    

def test_graph_printer():
    model_path = "raw_resnet18.onnx"
    
    graph = get_graph(model_path)
    printer = GraphPrinter()
    ret1 = printer(graph)
    
    mutator = GraphMutator()
    new_graph = mutator.mutate_graph(graph)
    ret2 = printer(new_graph)
    
    assert ret1 == ret2
    assert graph != new_graph
    
    
    
    
if __name__ == "__main__":
    test_basic_usage()
    test_graph_printer()