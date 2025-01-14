import argparse
import os
import tritonclient.grpc as grpcclient
#from utils import cal_cer
from speech_client import *
import numpy as np
import argparse
import time

#无法docker start直接打开是因为这是拉取的现有的镜像
#这个对象将用于定义和解析命令行参数。
parser = argparse.ArgumentParser() 
#出现-v或者verbose时需要设为true，非必需参数，默认false，用户使用help时提示 Enable verbose output允许详细输出调试信息
parser.add_argument(
    "-v",
    "--verbose",
    action="store_true",
    required=False,
    default=False,
    help="Enable verbose output",
)
#-u关注，默认非必要端口号，沟通服务端和客户端的
parser.add_argument(
    "-u",
    "--url",
    type=str,
    required=False,
    default="localhost:8001",     #这里被修改过,default="0.0.0.0:8001"只能在我的vscode终端中的venv_test虚拟环境中运行，在ubuntu的终端的venv_test虚拟环境中运行会报错,default="localhost:8001"可以在vscode中的终端的虚拟环境和ubuntu终端中的虚拟环境中运行
    # 报错为：Traceback (most recent call last):
#   File "/home/wangkeran/桌面/WENET/wenet/runtime/gpu/client/test.py", line 149, in <module>
#     result = speech_client.recognize(x)
#              ^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/wangkeran/桌面/WENET/wenet/runtime/gpu/client/speech_client.py", line 176, in recognize
#     response = self.triton_client.infer(
#                ^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/wangkeran/桌面/WENET/VENV_TEST/venv_test/lib/python3.12/site-packages/tritonclient/grpc/_client.py", line 1572, in infer
#     raise_error_grpc(rpc_error)
#   File "/home/wangkeran/桌面/WENET/VENV_TEST/venv_test/lib/python3.12/site-packages/tritonclient/grpc/_utils.py", line 77, in raise_error_grpc
#     raise get_error_grpc(rpc_error) from None
# tritonclient.utils.InferenceServerException: [StatusCode.UNAVAILABLE] failed to connect to all addresses; last error: UNAVAILABLE: ipv4:127.0.0.1:7897: Socket closed

    help="Inference server URL. Default is " "localhost:8001.",
)
#模型名称非必要，默认流式wenet，可以选择流式还是注意力式（非流式）
parser.add_argument(
    "--model_name",
    required=False,
    default="streaming_wenet",    #流式语音模型，这里修改过，
    #default="attention_rescoring",
    choices=["attention_rescoring", "streaming_wenet"],
    help="the model to send request to",
)
#音频路径非必要，默认没有音频，输入命令类型字符串，指定音频文件路径
parser.add_argument(
    "--wavscp",
    type=str,
    required=False,
    default=None,
    help="audio_id \t wav_path",
)
#非必须trans参数，用于控制模型的训练和解码过程中使用的Transformer结构，
#audio_id \t text 文件在wenet语音识别系统中起到了关键的桥梁作用，连接了语音数据与文本信息，为后续的处理步骤提供了必要的信息支持。
parser.add_argument(
    "--trans",
    type=str,
    required=False,
    default=None,
    help="audio_id \t text",
)
#音频数据位置，路径前缀
parser.add_argument(
    "--data_dir",
    type=str,
    required=False,
    default=None,
    help="path prefix for wav_path in wavscp/audio_file",
)
#音频文件，读取单个音频文件的路径
parser.add_argument(
    "--audio_file",
    type=str,
    required=False,
    default=None,
    help="single wav file path",
)
# below arguments are for streaming
# Please check onnx_config.yaml and train.yaml
#流式
parser.add_argument("--streaming", action="store_true", required=False)
#采样率，可选，默认16000赫兹，音频每秒取样16000次
parser.add_argument(
    "--sample_rate",
    type=int,
    required=False,
    default=16000,
    help="sample rate used in training",
)
#帧长度，默认25毫秒，帧是语音信号处理的基本单位，通常将连续的语音信号分割成一小段一小段，每小段称为一帧。这些帧之间通常会有交叠，以确保信息的连续性和完整性
parser.add_argument(
    "--frame_length_ms",
    type=int,
    required=False,
    default=25,
    help="frame length",
)
#帧移长度：每次滑动窗口偏移的样本点数或时间长度，单位毫秒，默认10毫秒
parser.add_argument(
    "--frame_shift_ms",
    type=int,
    required=False,
    default=10,
    help="frame shift length",
)
#块大小，单位是帧，决定每次处理块的大小，默认16帧，块是基于帧的更高层次的组织单位，通常用于限制注意力机制的作用范围。
parser.add_argument(
    "--chunk_size",
    type=int,
    required=False,
    default=16,
    help="chunk size default is 16",
)

#子采样上下文，默认单位是帧，25毫秒，共175毫秒。增强模型对语音上下文的理解和处理能力。
#具体来说，subsampling context通过将语音信号进行下采样处理，提取出更具有代表性的特征，从而帮助模型更好地捕捉语音信号中的关键信息和上下文关系。
parser.add_argument(
    "--context",
    type=int,
    required=False,
    default=7,
    help="subsampling context",
)
#下采样率：默认单位帧，通过降低音频信号的采样率来减少数据量，从而提高模型的计算效率和处理速度。
#具体来说，下采样率决定了音频信号在输入到模型之前被采样的频率。
#较高的下采样率可以更精确地表示高频信息，但会增加数据量和计算复杂度；
#而较低的下采样率则可以减少数据量，但可能会丢失一些高频信息。
#下采样率的选择应与语音识别服务支持的采样率保持一致，以确保识别效果不受影响
parser.add_argument(
    "--subsampling",
    type=int,
    required=False,
    default=4,
    help="subsampling rate",
)

start_time=time.time()
#解析从args[]传进来的命令
# FLAGS = parser.parse_args()
FLAGS = parser.parse_args(args=[])

#切块识别
speech_client_cls = StreamingSpeechClient
x="./test_wavs/mid.wav"    #这里是音频路径
#grpc创建连接
with grpcclient.InferenceServerClient(url=FLAGS.url, verbose=FLAGS.verbose) as triton_client:
    #赋值表示当前是GRPC连接
    protocol_client = grpcclient
    #使用 speech_client_cls 类初始化了一个 speech_client 对象。这个对象将使用 triton_client 进行推理，并且需要指定模型名称、协议客户端和一些其他参数。
    speech_client = speech_client_cls(triton_client, FLAGS.model_name, protocol_client, FLAGS)
    #可能是存储预测的预测参数
    predictions = []
    #语音识别
    result = speech_client.recognize(x)

end_time=time.time()
print(f"运行时间为：{end_time-start_time}秒")

#这里是输出的结果格式
#output
#Get response from 1th chunk: 
#Get response from 2th chunk: 大学
#Get response from 3th chunk: 大学生利用
#Get response from 4th chunk: 大学生利用漏洞
#Get response from 5th chunk: 大学生利用漏洞免费吃
#Get response from 6th chunk: 大学生利用漏洞免费吃肯德机
#Get response from 7th chunk: 大学生利用漏洞免费吃肯德机获刑
#Get response from 8th chunk: 大学生利用漏洞免费吃肯德基获刑

# 第一把的成绩
# 原流式运行时间为：0.17780041694641113秒,
# 原流式长音频：0.24332761764526367秒
# moonshine中音频：0.5764920711517334秒
# moonshine长音频：0.6331214904785156秒
# whisper-v3-turbo中音频：0.45749521255493164秒
# whisper-v3-turbo长音频：0.6579666137695312秒
# soyler超大模型中音频：0.6秒秒
# soyler超大模型长音频：0.5秒



#原版中音频稳定：0.11755776405334473秒
#原版长音频稳定：0.18992233276367188秒
# whisper-v3-turbo中音频稳定：0.11801719665527344秒
# whisper-v3-turbo长音频稳定：0.2638437747955322秒
# moonshine中音频稳定：0.16878366470336914秒
# moonshine长音频稳定：0.2508258819580078秒
# soyler超大模型中音频：0.12386918067932129秒秒
# soyler超大模型长音频：0.21893835067749023秒
# sherpa-onnx-zipformer穩定中音頻：0.1244194507598877秒
# sherpa-onnx-zipformer穩定长音频：0.22435283660888672秒