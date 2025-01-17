import onnx

# 加载模型,这里需要修改模型路径
model = onnx.load("/home/wangkeran/桌面/WENET/test-cpu-non-streaming-output/ctc.onnx")

# 查看输入信息
for input in model.graph.input:
    data_type = input.type.tensor_type.elem_type
    shape = input.type.tensor_type.shape
    print(f"Input Name: {input.name}")
    print(f"  Data Type: {onnx.TensorProto.DataType.Name(data_type)}")
    print(f"  Shape: {', '.join([dim.dim_value for dim in shape.dim]) if shape.dim else 'Unknown shape'}")

# 查看输出信息
for output in model.graph.output:
    data_type = output.type.tensor_type.elem_type
    shape = output.type.tensor_type.shape
    print(f"Output Name: {output.name}")
    print(f"  Data Type: {onnx.TensorProto.DataType.Name(data_type)}")
    print(f"  Shape: {', '.join([dim.dim_value for dim in shape.dim]) if shape.dim else 'Unknown shape'}")
