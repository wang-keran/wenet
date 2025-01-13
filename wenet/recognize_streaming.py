import soundfile as sf
import numpy as np
from wenet.wenet.transformer.asr_model import ASRModel
import sys
sys.path.insert(0, '/home/wangkeran/桌面/WENET/wenet')

# 初始化ASR模型
asr = Asr(lang='zh')  # 使用中文预训练模型

# 读取音频文件
audio_path = "/home/wangkeran/桌面/WENET/wenet/runtime/gpu/client/test_wavs/long.wav"  # 替换为你的音频文件路径
audio_data, sample_rate = sf.read(audio_path)

# 检查音频采样率
if sample_rate != 16000:
    raise ValueError(f"音频采样率应为16kHz，但当前音频采样率为 {sample_rate}Hz")

# 将音频数据转换为int16格式
audio_data = (audio_data * 32767).astype(np.int16)

# 每次传入的音频块大小
chunk_size = 1600  # 100ms的音频，16kHz采样率下1600帧

# 进行流式语音识别
print("开始读取音频并进行流式识别:")

for i in range(0, len(audio_data), chunk_size):
    chunk = audio_data[i:i + chunk_size]
    
    # 调用 WeNet 的 ASR 解码函数
    result = asr.decode(chunk)
    
    # 输出实时识别结果
    if result:
        print(f"识别结果: {result}")

print("识别结束")
