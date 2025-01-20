import onnx
from google.protobuf import text_format

#输出有问题，结构错误

# 加载 ONNX 模型
onnx_model = onnx.load("/home/wangkeran/桌面/WENET/test-cpu-non-streaming-output/decoder.onnx")

# 获取输入和输出的信息
inputs = onnx_model.graph.input
outputs = onnx_model.graph.output

# 打印输入和输出的名称及其形状和数据类型
print("Inputs:")
for input_tensor in inputs:
    print(f"Name: {input_tensor.name}, Shape: {input_tensor.type.tensor_type.shape}, DataType: {input_tensor.type.tensor_type.elem_type}")

print("\nOutputs:")
for output_tensor in outputs:
    print(f"Name: {output_tensor.name}, Shape: {output_tensor.type.tensor_type.shape}, DataType: {output_tensor.type.tensor_type.elem_type}")

# 将模型转换为 pbtxt 格式
onnx_model_pbtxt = text_format.MessageToString(onnx_model)

# 打印 pbtxt 格式的模型
print("\nONNX Model in pbtxt format:")
print(onnx_model_pbtxt)