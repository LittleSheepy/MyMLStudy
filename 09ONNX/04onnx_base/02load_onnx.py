import onnx



if __name__ == '__main__':
    torch_onnx = onnx.load("./torch.onnx")
    # check
    onnx.checker.check_model(torch_onnx)
    print(torch_onnx)