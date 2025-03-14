import onnx
from onnx import helper

print("加载onnx模型")
# 这里需要修改模型路径
model = onnx.load("/home/wangkeran/桌面/WENET/test-cpu-non-streaming-output/encoder.onnx")
print("加载完成，准备修改输入输出名称")
# 创建全0常量节点
offset_zero = helper.make_tensor(
    name="offset_zero",
    data_type=onnx.TensorProto.INT64,
    dims=[],
    vals=[0]
)
print("修改输入名称，准备创建常量节点")
const_node = helper.make_node(
    "Constant",
    inputs=[],
    outputs=["offset_zero"],
    value=offset_zero
)
print("修改完成，准备修改模型")
# 替换所有使用原offset输入的地方
for node in model.graph.node:
    if "offset" in node.input:  # 替换为实际offset输入名称
        node.input.remove("offset")
        node.input.append("offset_zero")
# 删除原offset输入节点
for i, input in enumerate(model.graph.input):
    if input.name == "offset":
        del model.graph.input[i]
        break
print("修改完成，准备保存模型")
onnx.save(model, "/home/wangkeran/桌面/WENET/test-cpu-non-streaming-output/encoder_fixed.onnx")