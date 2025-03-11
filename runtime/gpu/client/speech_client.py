# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from tritonclient.utils import np_to_triton_dtype
import numpy as np
import math
import soundfile as sf
import time


# 非流式客户端
class OfflineSpeechClient(object):

    # 初始化信息
    def __init__(self, triton_client, model_name, protocol_client, args):
        self.triton_client = triton_client
        self.protocol_client = protocol_client
        self.model_name = model_name

    # 识别音频文件
    def recognize(self, wav_file, idx=0):
        # 读取文件
        waveform, sample_rate = sf.read(wav_file)
        # 将读取的音频数据转换为适合模型输入的格式，并计算音频数据的长度。
        samples = np.array([waveform], dtype=np.float32)
        lengths = np.array([[len(waveform)]], dtype=np.int32)
        # better pad waveform to nearest length here
        # target_seconds = math.cel(len(waveform) / sample_rate)
        # target_samples = np.zeros([1, target_seconds  * sample_rate])
        # target_samples[0][0: len(waveform)] = waveform
        # samples = target_samples
        # 生成一个唯一的序列 ID，用于标识每个音频数据的处理请求。
        sequence_id = 10086 + idx
        result = ""
        # 创建并设置用于推理请求的输入数据
        inputs = [
            self.protocol_client.InferInput("WAV", samples.shape,
                                            np_to_triton_dtype(samples.dtype)),
            self.protocol_client.InferInput("WAV_LENS", lengths.shape,
                                            np_to_triton_dtype(lengths.dtype)),
        ]
        # 获取音频和长度
        inputs[0].set_data_from_numpy(samples)
        inputs[1].set_data_from_numpy(lengths)
        # 这行代码的作用是创建一个包含推理请求输出的列表，用于指定模型推理时需要返回的输出
        outputs = [self.protocol_client.InferRequestedOutput("TRANSCRIPTS")]
        # 给 Triton 推理服务器发送推理请求
        response = self.triton_client.infer(
            self.model_name,
            inputs,
            request_id=str(sequence_id),
            outputs=outputs,
        )
        # 获取推理结果
        decoding_results = response.as_numpy("TRANSCRIPTS")[0]
        # 这段代码的作用是处理推理结果，将其转换为字符串格式，并返回结果。如果结果是一个 NumPy 数组，则将其拼接成一个字符串；否则直接解码为字符串。
        if type(decoding_results) == np.ndarray:
            result = b" ".join(decoding_results).decode("utf-8")
        else:
            result = decoding_results.decode("utf-8")
        return [result]


# 流式客户端
class StreamingSpeechClient(object):

    # 初始化本方法内
    def __init__(self, triton_client, model_name, protocol_client, args):
        # 将传入的triton客户端，协议客户对象，模型名，协议对象名，其他指令
        self.triton_client = triton_client
        self.protocol_client = protocol_client
        self.model_name = model_name
        chunk_size = args.chunk_size
        subsampling = args.subsampling
        context = args.context
        # 帧移长度：每次滑动窗口偏移的样本点数或时间长度
        frame_shift_ms = args.frame_shift_ms
        # 帧长度
        frame_length_ms = args.frame_length_ms
        # for the first chunk
        # we need additional frames to generate
        # the exact first chunk length frames
        # since the subsampling will look ahead several frames
        # 第一帧长度：目的是为了在音频处理和语音识别任务中，确保每个音频块在解码时有足够的上下文信息。
        # 通过考虑下采样率和上下文帧数，可以确保解码过程的准确性和稳定性。
        first_chunk_length = (chunk_size - 1) * subsampling + context
        # 计算需要额外的帧数，以确保生成第一个块的精确长度。
        add_frames = math.ceil(
            (frame_length_ms - frame_shift_ms) / frame_shift_ms)
        # 计算第一个块的总时间（以毫秒为单位）。
        first_chunk_ms = (first_chunk_length + add_frames) * frame_shift_ms
        # 计算其他块的总时间（以毫秒为单位）。
        other_chunk_ms = chunk_size * subsampling * frame_shift_ms
        # 将第一个块的时间转换为秒。
        self.first_chunk_in_secs = first_chunk_ms / 1000
        # 将其他块的时间转换为秒。
        self.other_chunk_in_secs = other_chunk_ms / 1000

    def recognize(self, wav_file, idx=0):
        # 读取WAV文件，获取波形和采样率
        waveform, sample_rate = sf.read(wav_file)
        wav_segs = []
        i = 0
        # 分割音频波形存在wav_segs里
        # 这段代码的功能是将一个音频波形（waveform）分割成多个片段（wav_segs），
        # 根据指定的时间长度来确定每个片段的长度。
        while i < len(waveform):
            if i == 0:
                stride = int(self.first_chunk_in_secs * sample_rate)
                wav_segs.append(waveform[i:i + stride])
            else:
                stride = int(self.other_chunk_in_secs * sample_rate)
                wav_segs.append(waveform[i:i + stride])
            i += len(wav_segs[-1])

        # 设置结果id
        sequence_id = idx + 10086
        # simulate streaming，模拟流式语音识别，其实就是流式识别，但是是拿非流式模拟的
        # 将当前片段的数据填充到 expect_input 中，并将其转换为 numpy 数组。
        # 处理音频片段（wav_segs），并将它们逐个发送到 Triton 推理服务器进行推理，获取转录结果。
        for idx, seg in enumerate(wav_segs):
            # 计算片段长度
            chunk_len = len(seg)
            # 初始化输入数据
            if idx == 0:
                chunk_samples = int(self.first_chunk_in_secs * sample_rate)
                expect_input = np.zeros((1, chunk_samples), dtype=np.float32)
            else:
                chunk_samples = int(self.other_chunk_in_secs * sample_rate)
                expect_input = np.zeros((1, chunk_samples), dtype=np.float32)

            # 填充片段数据，每次添加一块新的数据和原来识别过的所有数据进去
            expect_input[0][0:chunk_len] = seg
            # 准备输入数据，这是音频数据
            input0_data = expect_input
            # 当前片段长度的数组
            input1_data = np.array([[chunk_len]], dtype=np.int32)

            # 设置输入和输出
            inputs = [
                self.protocol_client.InferInput(
                    "WAV",
                    input0_data.shape,
                    np_to_triton_dtype(input0_data.dtype),
                ),
                self.protocol_client.InferInput(
                    "WAV_LENS",
                    input1_data.shape,
                    np_to_triton_dtype(input1_data.dtype),
                ),
            ]

            # 使用 set_data_from_numpy 方法将 NumPy 数组数据设置到输入对象中。
            inputs[0].set_data_from_numpy(input0_data)
            inputs[1].set_data_from_numpy(input1_data)

            # 创建输出对象 outputs，指定需要返回的转录结果。
            outputs = [
                self.protocol_client.InferRequestedOutput("TRANSCRIPTS")
            ]
            # 根据当前索引 idx 判断是否是最后一个片段，若是，则设置 end = True，否则为 False。
            end = False
            if idx == len(wav_segs) - 1:
                end = True

            # 使用 self.triton_client.infer 方法向Triton推理服务器发送推理请求，并获取响应。
            response = self.triton_client.infer(
                self.model_name,
                inputs,
                outputs=outputs,
                sequence_id=sequence_id,
                sequence_start=idx == 0,
                sequence_end=end,
            )
            # 处理最后一个片段并返回
            idx += 1
            print("response的结果是：",response)
            # 获取响应结果，并通过 response.as_numpy("TRANSCRIPTS")[0].decode("utf-8") 解码出转录文本。
            time.sleep(1)
            result = response.as_numpy("TRANSCRIPTS")[0].decode("utf-8")
            # 输出结果，打印当前片段的索引和转录结果。
            print("Get response from {}th chunk: {}".format(idx, result))
        return [result]
