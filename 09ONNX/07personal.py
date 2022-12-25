import onnx
import onnxruntime
import numpy as np
import onnx.helper as helper

node = helper.make_node(
        "Mul_123",
        inputs=["input"],
        outputs=["output"],
        domain="ai.onnx.123",
    )

input = helper.make_tensor_value_info(
    name="input",
    elem_type=onnx.TensorProto.FLOAT,
    shape=[1,2]
)

output = helper.make_tensor_value_info(
    name="output",
    elem_type=onnx.TensorProto.FLOAT,
    shape=[1,2]
)

graph = helper.make_graph(
    nodes=[node],
    name="graph",
    inputs=[input],
    outputs=[output],
)

model = helper.make_model(graph)
opset=[]
opset.append(helper.make_operatorsetid(domain="ai.onnx.123",version=1))
onnx.checker.check_model(model, opset_import=opset)

