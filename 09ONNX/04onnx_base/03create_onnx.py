import onnx
import onnx.helper as helper
import numpy as np
input = helper.make_tensor_value_info(
    name="input",
    elem_type=onnx.TensorProto.FLOAT,
    shape=[1,3,224,224]
)

output = helper.make_tensor_value_info(
    name="output",
    elem_type=onnx.TensorProto.FLOAT,
    shape=[1,3,224,224]
)
weight = helper.make_tensor(
    name="weight",
    data_type=onnx.TensorProto.FLOAT,
    dims=[3,3,1,1],
    vals=np.random.randn(3,3,1,1)
)

bias = helper.make_tensor(
    name="bias",
    data_type=onnx.TensorProto.FLOAT,
    dims=[3],
    vals=np.random.randn(3)
)

node = helper.make_node(
    op_type="Conv",
    inputs=["input", "weight", "bias"],
    outputs=["output"],
    kernel_shape=[1,1],
    strides=[1,1],
    group=1,
    pads=[0,0,0,0],
)

graph = helper.make_graph(
    nodes=[node],
    name="graph",
    inputs=[input],
    outputs=[output],
    initializer=[weight, bias],
)

model = helper.make_model(graph)

onnx.checker.check_model(model)

onnx.save_model(model, "./model.onnx")
print(model)
