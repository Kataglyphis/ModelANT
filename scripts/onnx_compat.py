# Graph rewrites that keep an export's result and make it compute correctly on every execution provider.
import numpy as np
import onnx
from onnx import helper, numpy_helper

AXES = "kataglyphis_matmul_last_axis"


def _ranks(model):
    inferred = onnx.shape_inference.infer_shapes(model)
    ranks = {}
    graph = inferred.graph
    for value in list(graph.input) + list(graph.value_info) + list(graph.output):
        tensor = value.type.tensor_type
        if tensor.HasField("shape"):
            ranks[value.name] = len(tensor.shape.dim)
    ranks.update((init.name, len(init.dims)) for init in graph.initializer)
    for node in graph.node:
        if node.op_type == "Constant":
            for attribute in node.attribute:
                if attribute.name == "value":
                    ranks[node.output[0]] = len(attribute.t.dims)
    return ranks


def vector_matmuls(model):
    """The MatMul nodes that multiply a matrix by a 1-D vector."""
    ranks = _ranks(model)
    return [
        node
        for node in model.graph.node
        if node.op_type == "MatMul" and ranks.get(node.input[0], 0) >= 2 and ranks.get(node.input[1]) == 1
    ]


def expand_vector_matmuls(model):
    """Write MatMul(matrix, vector) as a matrix product of a column; returns how many nodes changed.

    DirectML computes the 1-D form wrongly: D-FINE's box integral came out garbage on AMD GPUs with
    every graph optimization level, and detections vanished while the provider profile looked fine.
    """
    targets = {id(node) for node in vector_matmuls(model)}
    if not targets:
        return 0
    model.graph.initializer.append(numpy_helper.from_array(np.array([-1], dtype=np.int64), AXES))
    nodes = []
    for node in model.graph.node:
        if id(node) not in targets:
            nodes.append(node)
            continue
        column, product = f"{node.output[0]}/column", f"{node.output[0]}/product"
        nodes.append(helper.make_node("Unsqueeze", [node.input[1], AXES], [column], name=f"{node.name}/Unsqueeze"))
        nodes.append(helper.make_node("MatMul", [node.input[0], column], [product], name=node.name))
        nodes.append(helper.make_node("Squeeze", [product, AXES], [node.output[0]], name=f"{node.name}/Squeeze"))
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    return len(targets)
