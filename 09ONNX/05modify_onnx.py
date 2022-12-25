import torch
import torch.nn as nn
import onnxruntime
import onnx
import numpy as np
import onnx.helper as helper

def add_node():
    model = onnx.load("./model.onnx")
    nodes = model.graph.node
    new_node = helper.make_node(
        "Relu",
        inputs=['conv1'],
        outputs=['output'],
        name='relu1',
    )

    nodes.append(new_node)
    nodes[0].output[0] = "conv1"
    onnx.checker.check_model(model)
    onnx.save_model(model, "./add_model.onnx")
    input = np.random.rand(1,3,224,224).astype(dtype=np.float32)
    sess = onnxruntime.InferenceSession("./add_model.onnx")
    result = sess.run(["output"],{"input":input})
    print(result) # [shape=1,3,224,224]

def del_node():
    model = onnx.load("./add_model.onnx")
    nodes = model.graph.node
    for node in nodes:
        if node.name == "relu1":
            print(node)
            nodes.remove(node)
    nodes[0].output[0] = "output"
    onnx.checker.check_model(model)
    onnx.save_model(model, "./del_model.onnx")
    input = np.random.rand(1,3,224,224).astype(dtype=np.float32)
    sess = onnxruntime.InferenceSession("./del_model.onnx")
    result = sess.run(["output"],{"input":input})
    print(result) # [shape=1,3,224,224]

del_node()