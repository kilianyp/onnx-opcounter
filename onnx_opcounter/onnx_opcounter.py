import logging
import collections
from onnx import shape_inference
import onnx
from onnx import numpy_helper


logger = logging.getLogger(__name__)


def calculate_params(model: onnx.ModelProto) -> int:
    onnx_weights = model.graph.initializer
    params = 0

    for onnx_w in onnx_weights:
        try:
            weight = numpy_helper.to_array(onnx_w)
            params += np.prod(weight.shape)
        except Exception as _:
            pass
    return params


def to_list(shape_proto):
    shape = []
    for dim in shape_proto.dim:
        shape.append(dim.dim_value)
    return shape


import onnxruntime as rt
import numpy as np

def calculate_macs(model: onnx.ModelProto):
    print(1)
    model = shape_inference.infer_shapes(model)


    graph_weights = [w.name for w in model.graph.initializer]
    graph_outputs = set(i.name for i in model.graph.output)
    for node in model.graph.node:
        if node.name in graph_outputs:
            continue
        intermediate_layer_value_info = onnx.helper.ValueInfoProto()
        intermediate_layer_value_info.name = node.name
        model.graph.output.extend([intermediate_layer_value_info])
        graph_outputs.add(node.name)

    onnx.save(model, '+all-intermediate.onnx')
    sess = rt.InferenceSession('+all-intermediate.onnx')

    # construc
    input_sample = {}
    type_mapping = {
        1: np.float32,
        7: np.int64,
    }

    for graph_input in model.graph.input:
        if graph_input.name not in graph_weights:
            input_sample[graph_input.name] = \
                np.zeros([i.dim_value for i in graph_input.type.tensor_type.shape.dim],
                         dtype=type_mapping[graph_input.type.tensor_type.elem_type])
    output = sess.run(graph_outputs, input_sample)
    return output
    print(output)

    shapes = {}
    for v in model.graph.value_info:
        shapes[v.name] = to_list(v.type.tensor_type.shape)


    onnx_nodes = model.graph.node
    onnx_weights = model.graph.initializer

    for w in onnx_weights:
        shapes[w.name] = w.dims

    for o in model.graph.output:
        shapes[o.name] = to_list(o.type.tensor_type.shape)

    for i in model.graph.input:
        shapes[i.name] = to_list(i.type.tensor_type.shape)



    counter = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
    for node in onnx_nodes:
        try:
            input_shapes = [tuple(shapes[i]) for i in node.input]
            output_shapes = [tuple(shapes[o]) for o in node.output]
            in_out_shapes = tuple(input_shapes + [None] + output_shapes)
            counter[node.op_type][in_out_shapes] += 1
        except:
            print(f"could not deal with {node.name}")

    return counter
