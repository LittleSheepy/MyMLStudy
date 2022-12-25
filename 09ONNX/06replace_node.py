import onnx
import onnxruntime
import numpy as np
import onnx.helper as helper


def replace():
    model = onnx.load("./add_model.onnx")
    nodes = model.graph.node
    new_node = helper.make_node(
        "Squeeze",
        inputs=["input"],
        outputs=["conv1"],
        name='squeeze1',
    )
    #nodes.append(new_node)
    for idx, node in enumerate(nodes):
        if node.op_type == "Conv":
            nodes.remove(node)
            nodes.insert(idx, new_node)

    onnx.checker.check_model(model)
    onnx.save_model(model, "./replace_model.onnx")
    input = np.random.rand(1,3,224,224).astype(dtype=np.float32)
    sess = onnxruntime.InferenceSession("./replace_model.onnx")
    result = sess.run(["output"],{"input":input})
    print(result) # [shape=1,3,224,224]




replace()







