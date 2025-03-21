import onnx
import os

# 设置模型路径
#model_path = "/home/wangkeran/桌面/WENET/test-cpu-non-streaming-output/encoder.onnx"
model_path = "/home/wangkeran/桌面/WENET/aishellout-streaming/encoder.onnx"
#model_path = "/home/wangkeran/桌面/WENET/aishellout-non-streaming/encoder.onnx"
print("模型路径：",model_path)

# 检查模型文件是否存在
if not os.path.exists(model_path):
    print(f"Error: The model file at {model_path} does not exist.")
else:
    # 加载模型
    try:
        model = onnx.load(model_path)

        # 查看输入信息
        print("Inputs:")
        for input in model.graph.input:
            data_type = input.type.tensor_type.elem_type
            shape = input.type.tensor_type.shape
            shape_str = ', '.join([str(dim.dim_value) for dim in shape.dim]) if shape.dim else 'Unknown shape'
            print(f"  Input Name: {input.name}")
            print(f"    Data Type: {onnx.TensorProto.DataType.Name(data_type)}")
            print(f"    Shape: {shape_str}")

        # 查看输出信息
        print("\nOutputs:")
        for output in model.graph.output:
            data_type = output.type.tensor_type.elem_type
            shape = output.type.tensor_type.shape
            shape_str = ', '.join([str(dim.dim_value) for dim in shape.dim]) if shape.dim else 'Unknown shape'
            print(f"  Output Name: {output.name}")
            print(f"    Data Type: {onnx.TensorProto.DataType.Name(data_type)}")
            print(f"    Shape: {shape_str}")

    except Exception as e:
        print(f"Error loading the ONNX model: {e}")