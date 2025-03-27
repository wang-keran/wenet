import onnxruntime as ort

#model_path = "/home/wangkeran/桌面/WENET/gpu_cpu_non_streaming/decoder_fp16.onnx"  # 你的 ONNX 模型路径
model_path = "/home/wangkeran/桌面/WENET/aishellout/decoder_fp16.onnx"  # 你的 ONNX 模型路径

session = ort.InferenceSession(model_path)

# 获取当前使用的 Provider（执行设备）
providers = session.get_providers()
print("Current Execution Provider:", providers)