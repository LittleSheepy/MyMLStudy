import onnx
import onnxruntime
import numpy as np

model = onnx.load("./dynamic_model.onnx")
print(model)
#import pdb;pdb.set_trace()
# model.graph.input[0].type.tensor_type.shape.dim[0].dim_value
input = model.graph.input
output = model.graph.output
# for i in input:
#     i.type.tensor_type.shape.dim[0].dim_value = 2
# for i in output:
#     i.type.tensor_type.shape.dim[0].dim_value = 2


for i in input:
    i.type.tensor_type.shape.dim[0].dim_param = "batchsize"
for i in output:
    i.type.tensor_type.shape.dim[0].dim_param = "batchsize"
onnx.checker.check_model(model)
onnx.save_model(model, "./dynamic_model.onnx")
sess = onnxruntime.InferenceSession("dynamic_model.onnx")
input = np.random.rand(16,3,224,224).astype(dtype=np.float32)
result = sess.run(["output"],{"input":input})
print(result) # [shape=1,3,224,224]