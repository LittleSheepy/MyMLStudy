import torch
import torch.nn as nn
from MultiHeadedAttention.MultiHeadedAttention01 import MultiHeadAttention as MultiHeadAttention01
from MultiHeadedAttention.MultiHeadAttention02 import MultiHeadAttention as MultiHeadAttention02
import onnx


if __name__ == '__main__':
    model = MultiHeadAttention02(8, 512)
    #model2 = WindowAttention_v2(32,(7,7), 2)
    #dummy = torch.zeros(1, 32, 128, 128)
    # torch.Size([361, 49, 32])

    dummy = torch.zeros(16, 50, 512)
    onnx_name = "D:/000/MultiHeadAttention.onnx"
    onnx_model = torch.onnx.export(model, (dummy,),
                      onnx_name,
                      opset_version=11)
    # onnx_model = onnx.load(onnx_name)
    # graph = onnx_model.graph
    # nodes = graph.node
    # inputs = graph.input
    # outputs = graph.output
