import onnx
import onnxruntime as rt
import numpy as np
from onnx import numpy_helper
import ipdb


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


def onnx_node_attributes_to_dict(args):
    """
    Parse ONNX attributes to Python dictionary
    :param args: ONNX attributes object
    :return: Python dictionary
    """

    def onnx_attribute_to_dict(onnx_attr):
        """
        Parse ONNX attribute
        :param onnx_attr: ONNX attribute
        :return: Python data type
        """
        if onnx_attr.HasField('t'):
            return numpy_helper.to_array(getattr(onnx_attr, 't'))

        for attr_type in ['f', 'i', 's']:
            if onnx_attr.HasField(attr_type):
                return getattr(onnx_attr, attr_type)

        for attr_type in ['floats', 'ints', 'strings']:
            if getattr(onnx_attr, attr_type):
                return list(getattr(onnx_attr, attr_type))

    return {arg.name: onnx_attribute_to_dict(arg) for arg in args}


from onnx import shape_inference
def to_list(shape_proto):
    shape = []
    for dim in shape_proto.dim:
        shape.append(dim.dim_value)
    return shape


def calculate_macs(model: onnx.ModelProto) -> int:
    if len(model.graph.value_info) == 0:
        model = shape_inference.infer_shapes(model)

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


    import collections
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
